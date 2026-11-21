"""Hook contract fixtures only: no mock result is evidence of ASAP physics."""
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))

from rank_hook import EventLog, installed_hooks, json_value


class FixtureGenerator:
    """Models only the audited pre-yield RNG boundary, not a CAD generator."""
    base_part = "6"

    def __init__(self):
        self.calls = 0
        self.native_yields = []

    def generate_candidate_part(self, assembled_parts):
        self.calls += 1
        sorted_parts = [part for part in assembled_parts if part != self.base_part]
        random.shuffle(sorted_parts)
        for part in sorted_parts:
            self.native_yields.append(part)
            yield part


class FixturePlanner:
    """Returns local test data; never used by the actual worker launcher."""
    def __init__(self):
        self.raw = {"feasible": False, "details": {"observations": ["test-only"]}}
        self.error = None
        self.calls = 0

    def _simulate(self, part_move, parts_rest, parts_removed, pose, max_grippers,
                  timeout=None, optimizer=None, debug=False, render=False):
        self.calls += 1
        if self.error is not None:
            raise self.error
        random.random()
        return self.raw


def rng_digest():
    return hashlib.sha256(repr(random.getstate()).encode("ascii")).hexdigest()


class RankHookTests(unittest.TestCase):
    def setUp(self):
        self.previous_rng = random.getstate()
        self.directory = tempfile.TemporaryDirectory()
        self.trace_path = Path(self.directory.name) / "events.jsonl"
        self.log = EventLog(self.trace_path, "contract-run")

    def tearDown(self):
        self.log.close()
        self.directory.cleanup()
        random.setstate(self.previous_rng)

    def events(self):
        return [json.loads(line) for line in self.trace_path.read_text(
            encoding="utf-8").splitlines()]

    def run_prefix(self, mode, path):
        random.seed(41)
        log = EventLog(path, "same-test-run")
        generator, planner = FixtureGenerator(), FixturePlanner()
        parts = ["1", "0", "3", "2", "6"]
        consumed = []
        try:
            with installed_hooks(FixtureGenerator, FixturePlanner, log, mode, rng_digest):
                iterator = generator.generate_candidate_part(parts)
                for _ in range(2):
                    candidate = next(iterator)
                    consumed.append(candidate)
                    returned = planner._simulate(
                        candidate, [part for part in parts if part != candidate],
                        [], [[1.0, 0.0], [0.0, 1.0]], 2)
                    self.assertIs(returned, planner.raw)
                iterator.close()
        finally:
            log.close()
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return events, consumed, rng_digest(), generator

    def test_trace_and_noop_match_consumed_prefix_rng_and_verifier_events(self):
        directory = Path(self.directory.name)
        trace, trace_parts, trace_rng, trace_generator = self.run_prefix(
            "trace", directory / "trace.jsonl")
        noop, noop_parts, noop_rng, noop_generator = self.run_prefix(
            "noop", directory / "noop.jsonl")

        def observed(events):
            return [{key: value for key, value in event.items() if key != "event_id"}
                    for event in events if event["kind"] != "rank_decision"]

        self.assertEqual(observed(trace), observed(noop))
        self.assertEqual(trace_parts, noop_parts)
        self.assertEqual(trace_rng, noop_rng)
        self.assertEqual(trace_generator.calls, 1)
        self.assertEqual(noop_generator.calls, 1)
        # Trace remains lazy. Eager noop is safe only for the audited generator.
        self.assertEqual(len(trace_generator.native_yields), 2)
        self.assertEqual(len(noop_generator.native_yields), 4)
        generation = next(event for event in noop if event["kind"] == "generation")
        self.assertEqual(generation["native_order"], generation["applied_order"])
        self.assertEqual(len(generation["native_order"]), 4)
        self.assertEqual([event["part"] for event in noop
                          if event["kind"] == "candidate_yield"], noop_parts)
        rank = next(event for event in noop if event["kind"] == "rank_decision")
        self.assertIsNone(rank["fallback_reason"])
        self.assertEqual(rank["request"]["candidate_ids"], generation["native_order"])

    def test_empty_native_iterator_is_recorded_in_both_modes(self):
        for mode in ("trace", "noop"):
            with self.subTest(mode=mode):
                with installed_hooks(FixtureGenerator, FixturePlanner, self.log, mode, rng_digest):
                    self.assertEqual(list(FixtureGenerator().generate_candidate_part(["6"])), [])
        events = self.events()
        generations = [event for event in events if event["kind"] == "generation"]
        self.assertEqual(len(generations), 2)
        for event in generations:
            self.assertEqual(event["native_order"], [])
            self.assertEqual(event["applied_order"], [])
        self.assertFalse(any(event["kind"] == "candidate_yield" for event in events))

    def test_original_class_methods_are_restored_after_success(self):
        original_generate = FixtureGenerator.generate_candidate_part
        original_simulate = FixturePlanner._simulate
        with installed_hooks(FixtureGenerator, FixturePlanner, self.log, "noop", rng_digest):
            self.assertIsNot(FixtureGenerator.generate_candidate_part, original_generate)
            self.assertIsNot(FixturePlanner._simulate, original_simulate)
            list(FixtureGenerator().generate_candidate_part(["1", "6"]))
        self.assertIs(FixtureGenerator.generate_candidate_part, original_generate)
        self.assertIs(FixturePlanner._simulate, original_simulate)

    def test_simulation_exception_is_preserved_and_class_patches_are_restored(self):
        original_generate = FixtureGenerator.generate_candidate_part
        original_simulate = FixturePlanner._simulate
        planner = FixturePlanner()
        planner.error = ValueError("fixture exception")
        with self.assertRaises(ValueError) as caught:
            with installed_hooks(FixtureGenerator, FixturePlanner, self.log, "noop", rng_digest):
                planner._simulate("1", ["6"], [], None, 2)
        self.assertIs(caught.exception, planner.error)
        self.assertIs(FixtureGenerator.generate_candidate_part, original_generate)
        self.assertIs(FixturePlanner._simulate, original_simulate)
        result = self.events()[-1]
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["error_type"], "ValueError")
        self.assertIsNone(result["raw"])
        self.assertIsNone(result["blocker"])
        self.assertIsNone(result["path"])

    def test_native_generator_exception_is_not_relabelled_as_ranker_fallback(self):
        class FailingGenerator:
            base_part = "6"

            def generate_candidate_part(self, assembled_parts):
                raise RuntimeError("native fixture failed")
                yield  # pragma: no cover -- makes this an iterator fixture

        original = FailingGenerator.generate_candidate_part
        for mode in ("trace", "noop"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(RuntimeError, "native fixture failed"):
                    with installed_hooks(FailingGenerator, FixturePlanner, self.log,
                                         mode, rng_digest):
                        list(FailingGenerator().generate_candidate_part(["1", "6"]))
                self.assertIs(FailingGenerator.generate_candidate_part, original)
        self.assertEqual(self.events(), [])

    def test_false_result_stays_unknown_and_raw_snapshot_cannot_change_later(self):
        planner = FixturePlanner()
        with installed_hooks(FixtureGenerator, FixturePlanner, self.log, "noop", rng_digest):
            returned = planner._simulate("1", ["6"], [], None, 2)
            self.assertIs(returned, planner.raw)
            returned["details"]["observations"].append("later mutation")
        result = self.events()[-1]
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIs(result["raw"]["feasible"], False)
        self.assertEqual(result["raw"]["details"]["observations"], ["test-only"])
        self.assertIsNone(result["blocker"])
        self.assertIsNone(result["path"])

    def test_only_explicit_true_is_pass(self):
        for feasible, expected in ((True, "PASS"), (False, "UNKNOWN"),
                                   (None, "UNKNOWN"), (1, "UNKNOWN")):
            with self.subTest(feasible=feasible):
                planner = FixturePlanner()
                planner.raw = {"feasible": feasible}
                with installed_hooks(FixtureGenerator, FixturePlanner, self.log,
                                     "noop", rng_digest):
                    planner._simulate("1", ["6"], [], None, 2)
                self.assertEqual(self.events()[-1]["status"], expected)

    def test_leaf_validation_has_no_stale_generation_link(self):
        generator, planner = FixtureGenerator(), FixturePlanner()
        with installed_hooks(FixtureGenerator, FixturePlanner, self.log, "noop", rng_digest):
            iterator = generator.generate_candidate_part(["1", "2", "3", "6"])
            candidate = next(iterator)
            rest = [part for part in ["1", "2", "3", "6"] if part != candidate]
            planner._simulate(candidate, rest, [], None, 2)
            planner._simulate("2", ["6"], ["1", "3"], None, 2)
            iterator.close()
        starts = [event for event in self.events() if event["kind"] == "verifier_start"]
        self.assertEqual(starts[0]["generation_id"], "g1")
        self.assertIsNone(starts[1]["generation_id"])

    def test_rank_request_and_event_ids_are_unique_across_generations(self):
        with installed_hooks(FixtureGenerator, FixturePlanner, self.log, "noop", rng_digest):
            generator = FixtureGenerator()
            list(generator.generate_candidate_part(["1", "6"]))
            list(generator.generate_candidate_part(["1", "6"]))
        events = self.events()
        ids = [event["event_id"] for event in events]
        self.assertEqual(len(ids), len(set(ids)))
        ranks = [event for event in events if event["kind"] == "rank_decision"]
        self.assertNotEqual(ranks[0]["request"]["request_id"], ranks[1]["request"]["request_id"])
        self.assertEqual(ranks[0]["request"]["state_id"], ranks[1]["request"]["state_id"])

    def test_existing_event_file_is_never_overwritten(self):
        self.log.emit("existing", value="preserve")
        before = self.trace_path.read_bytes()
        with self.assertRaises(FileExistsError):
            EventLog(self.trace_path, "second-run")
        self.assertEqual(self.trace_path.read_bytes(), before)

    def test_json_snapshot_copies_array_like_values_and_rejects_unknown_objects(self):
        class ArrayLike:
            def __init__(self):
                self.values = [[1.0, 2.0]]

            def tolist(self):
                return self.values

        original = ArrayLike()
        copied = json_value({"pose": original})
        original.values[0][0] = 9.0
        self.assertEqual(copied, {"pose": [[1.0, 2.0]]})
        with self.assertRaises(TypeError):
            json_value(object())


if __name__ == "__main__":
    unittest.main()
