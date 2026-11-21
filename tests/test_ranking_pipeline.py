"""Offline composition fixtures; no API, WSL, or physical validation is run."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orchestrator.ranking_pipeline import RankingPipeline
from evidence_memory import POLICY, beam_context, canonical, digest, record_from_pair
from memory_ranker import BEAM_SETUP, MemoryRanker
from rank_contract import RankRequest


def context_fixture():
    return beam_context({"commit": "fixture-only", "asset_hash": "mesh",
                         "config_hash": "config", "binary_hash": "binary",
                         "sdf_hash": "sdf"}, BEAM_SETUP, seed=0)


def request_fixture(tag="r1", candidates=None, timeout=45):
    return {"schema_version": 1, "kind": "rank_request", "request": {
        "run_id": "fixture", "request_id": "fixture:" + tag, "state_id": "s0",
        "assembled_parts": ["0", "1", "2", "3", "6"],
        "candidate_ids": ["2", "3", "0", "1"] if candidates is None else candidates,
        "base_part": "6"}, "features": {"geometry": {"frame": "fixture"}},
        "timeout_seconds": timeout}


def config_fixture(**changes):
    return {"enabled": True, "model": "gpt-6-luna", "api_key_env": "PIPELINE_TEST_KEY",
            "max_calls": 10, "max_usd": "10", "input_usd_per_million": "2",
            "output_usd_per_million": "4", **changes}


def provider_response(request, candidates=None):
    value = {"request_id": request["request_id"], "state_id": request["state_id"],
             "candidate_ids": candidates if candidates is not None else request["candidate_ids"]}
    return {"id": "response_fixture", "model": "fixture-returned-model", "status": "completed",
            "error": None, "output": [{"type": "message", "role": "assistant",
                "status": "completed", "content": [{"type": "output_text", "text": json.dumps(value)}]}],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}


def evidence(context, part, status, tag):
    request = dict(part_move=part, parts_rest=[p for p in context["initial_parts"] if p != part],
                   parts_removed=[], pose=None, max_grippers=2, timeout=None,
                   optimizer="L-BFGS-B", debug=1, render=False)
    start = dict(kind="verifier_start", run_id="development", event_id=tag + ":start",
                 call_id=tag, request=request)
    result = dict(kind="verifier_result", run_id="development", event_id=tag + ":result",
                  call_id=tag, status=status, error_type="FixtureError" if status == "ERROR" else None,
                  raw=None if status == "ERROR" else dict(
                      feasible=status == "PASS", part_move=part, base_part="6", pose=None,
                      action=[0, 0, 1] if status == "PASS" else None,
                      parts_fix=[] if status == "PASS" else None, grasp=None))
    return record_from_pair(context, start, result, "synthetic/development.jsonl", "a" * 64)


class RankingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="asap_pipeline_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.context = context_fixture()
        self.snapshot = self.root / "memory.json"
        records = [evidence(self.context, part, status, str(i)) for i, (part, status) in
                   enumerate((("0", "PASS"), ("2", "UNKNOWN"), ("3", "ERROR")))]
        self.write_snapshot(records)
        self.serial = 0
        env = patch.dict(os.environ, {"PIPELINE_TEST_KEY": "offline-fixture-key"})
        env.start()
        self.addCleanup(env.stop)
        network = patch("urllib.request.OpenerDirector.open",
                        side_effect=AssertionError("network_forbidden_in_fixture"))
        network_mock = network.start()
        self.addCleanup(network.stop)
        self.addCleanup(network_mock.assert_not_called)

    def write_snapshot(self, records):
        records = sorted(records, key=lambda item: item["record_id"])
        value = dict(schema_version=1, records=records, record_count=len(records),
                     source_record_ids=[record["record_id"] for record in records],
                     content_sha256=digest(records), policy=POLICY, allow_hard_pruning=False)
        self.snapshot.write_text(canonical(value), encoding="utf-8")

    def pipeline(self, mode="on", base="noop", cfg=None, transport=None, context=None,
                 snapshot=None, ledger=None):
        self.serial += 1
        return RankingPipeline(
            config_fixture(enabled=False) if cfg is None else cfg,
            ledger or self.root / ("ledger%d.sqlite" % self.serial),
            self.snapshot if snapshot is None else snapshot,
            self.context if context is None else context, mode, base, transport)

    def transport(self, candidates=None):
        return Mock(side_effect=lambda url, headers, body, timeout: provider_response(
            json.loads(body["input"][0]["content"])["request"], candidates))

    def assert_response(self, result, candidates, reason, source="deterministic_fallback"):
        self.assertEqual(set(result), {"schema_version", "kind", "request_id", "state_id",
                                      "candidate_ids", "fallback_reason", "metadata"})
        self.assertEqual(result["kind"], "rank_response")
        self.assertEqual(result["candidate_ids"], candidates)
        self.assertIsNone(result["fallback_reason"])
        self.assertEqual(result["metadata"]["llm_fallback_reason"], reason)
        self.assertEqual(result["metadata"]["ranking_source"], source)
        self.assertEqual(result["metadata"]["model_requested"], "gpt-6-luna")

    def test_off_does_not_open_or_query_evidence(self):
        with patch("orchestrator.ranking_pipeline.MemoryRanker") as ranker:
            result = self.pipeline("off", snapshot=self.root / "absent.json").rank(request_fixture())
        ranker.assert_not_called()
        self.assert_response(result, ["2", "3", "0", "1"], "llm_disabled")
        self.assertFalse(result["metadata"]["memory"]["evidence_loaded"])
        self.assertIsNone(result["metadata"]["memory"]["snapshot_sha256"])

    def test_disabled_memory_on_matches_original_memory_ranker(self):
        offered = request_fixture()
        request = offered["request"]
        original = RankRequest(request["run_id"], request["request_id"], request["state_id"],
                               tuple(request["assembled_parts"]), tuple(request["candidate_ids"]), "6")
        order, metadata = MemoryRanker(self.snapshot, self.context, "on").apply(
            original, original.candidate_ids)
        result = self.pipeline().rank(offered)
        self.assert_response(result, list(order), "llm_disabled")
        self.assertEqual(list(order), ["0", "2", "3", "1"])
        for name, value in metadata.items():
            self.assertEqual(result["metadata"]["memory"][name], value)
        self.assertEqual(result["metadata"]["memory"]["status_counts"],
                         {"PASS": 1, "UNKNOWN": 1, "ERROR": 1})
        self.assertFalse(result["metadata"]["llm_live"])

    def test_reverse_base_and_partial_candidates_never_add_or_remove_ids(self):
        off = self.pipeline("off", "reverse_native").rank(request_fixture())
        on = self.pipeline("on", "reverse_native").rank(request_fixture())
        self.assert_response(off, ["1", "0", "3", "2"], "llm_disabled")
        self.assert_response(on, ["0", "1", "3", "2"], "llm_disabled")
        partial = self.pipeline().rank(request_fixture(candidates=["2", "1"]))
        self.assert_response(partial, ["2", "1"], "llm_disabled")
        self.assertEqual(partial["metadata"]["memory"]["positive_candidate_ids"], ["0"])
        self.assertEqual(partial["metadata"]["memory"]["offered_positive_candidate_ids"], [])
        empty = self.pipeline().rank(request_fixture(candidates=[]))
        self.assert_response(empty, [], "llm_disabled")

    def test_unknown_and_error_only_never_penalize_candidates(self):
        self.write_snapshot([evidence(self.context, "2", "UNKNOWN", "u"),
                             evidence(self.context, "3", "ERROR", "e")])
        result = self.pipeline().rank(request_fixture())
        self.assert_response(result, ["2", "3", "0", "1"], "llm_disabled")
        self.assertEqual(result["metadata"]["memory"]["positive_candidate_ids"], [])
        self.assertFalse(result["metadata"]["memory"]["changed"])

    def test_exact_context_seed_input_binary_setup_changes_do_not_hit(self):
        for label in ("seed", "asset_hash", "binary_hash", "setup"):
            changed = deepcopy(self.context)
            if label == "seed":
                changed["seed"] = 1
            elif label == "setup":
                changed["setup"]["budget"] = 399
            else:
                changed["input_snapshot"][label] = "changed"
            with self.subTest(label=label):
                result = self.pipeline(context=changed).rank(request_fixture())
                self.assert_response(result, ["2", "3", "0", "1"], "llm_disabled")
                self.assertEqual(result["metadata"]["memory"]["matched_evidence_ids"], [])
                self.assertEqual(result["metadata"]["memory"]["context_digest"], digest(changed))

    def test_enabled_service_gets_memory_hints_but_uses_valid_llm_permutation(self):
        proposed = ["3", "1", "2", "0"]
        transport = self.transport(proposed)
        offered = request_fixture()
        saved = deepcopy(offered)
        result = self.pipeline(cfg=config_fixture(), transport=transport).rank(offered)
        self.assert_response(result, proposed, None, "llm")
        transmitted = json.loads(transport.call_args.args[2]["input"][0]["content"])
        self.assertEqual(transmitted["request"], offered["request"])
        self.assertEqual(transmitted["features"]["deterministic_fallback_order"], ["0", "2", "3", "1"])
        self.assertEqual(transmitted["features"]["memory"], result["metadata"]["memory"])
        self.assertFalse(transmitted["features"]["memory"]["allow_hard_pruning"])
        self.assertIn("UNKNOWN and ERROR never", transmitted["features"]["memory"]["hint_scope"])
        self.assertEqual(offered, saved)
        self.assertFalse(result["metadata"]["llm_live"])
        self.assertTrue(result["metadata"]["attempted"])
        self.assertEqual(result["metadata"]["model_returned"], "fixture-returned-model")
        self.assertEqual(result["metadata"]["usage"]["total_tokens"], 15)

    def test_api_error_timeout_and_invalid_permutation_use_positive_fallback(self):
        cases = ((Mock(side_effect=RuntimeError("sensitive provider body")), "api_error"),
                 (Mock(side_effect=TimeoutError("fixture")), "api_timeout"),
                 (self.transport(["0"]), "invalid_permutation"))
        for transport, reason in cases:
            with self.subTest(reason=reason):
                result = self.pipeline(cfg=config_fixture(), transport=transport).rank(request_fixture())
                self.assert_response(result, ["0", "2", "3", "1"], reason)
                self.assertNotIn("sensitive provider body", canonical(result))

    def test_off_on_send_same_candidates_with_explicit_evidence_difference(self):
        sent = {}
        for mode in ("off", "on"):
            transport = self.transport()
            self.pipeline(mode, cfg=config_fixture(), transport=transport).rank(request_fixture())
            sent[mode] = json.loads(transport.call_args.args[2]["input"][0]["content"])
        self.assertEqual(sent["off"]["request"], sent["on"]["request"])
        self.assertEqual(sent["off"]["features"]["geometry"], sent["on"]["features"]["geometry"])
        self.assertEqual(sent["off"]["features"]["memory"]["matched_evidence_ids"], [])
        self.assertEqual(sent["off"]["features"]["memory"]["status_counts"], {})
        self.assertEqual(sent["off"]["features"]["deterministic_fallback_order"], ["2", "3", "0", "1"])
        self.assertEqual(sent["on"]["features"]["deterministic_fallback_order"], ["0", "2", "3", "1"])
        self.assertEqual(len(sent["on"]["features"]["memory"]["matched_evidence_ids"]), 3)

    def test_budget_exhaustion_uses_positive_fallback_without_new_transport(self):
        transport = self.transport()
        pipeline = self.pipeline(cfg=config_fixture(max_calls=1), transport=transport)
        pipeline.rank(request_fixture("first"))
        second = pipeline.rank(request_fixture("second"))
        self.assert_response(second, ["0", "2", "3", "1"], "call_budget_exhausted")
        self.assertEqual(transport.call_count, 1)

    def test_invalid_memory_is_type_only_and_does_not_break_base_order(self):
        self.snapshot.write_text('{"schema_version":0}', encoding="utf-8")
        result = self.pipeline(base="reverse_native").rank(request_fixture())
        self.assert_response(result, ["1", "0", "3", "2"], "llm_disabled")
        self.assertEqual(result["metadata"]["memory"]["fallback_reason"], "ValueError")
        self.assertFalse(result["metadata"]["memory"]["evidence_loaded"])
        pipeline = self.pipeline(snapshot=self.root / "missing.json")
        result = pipeline.rank(request_fixture())
        self.assertEqual(result["metadata"]["memory"]["fallback_reason"], "FileNotFoundError")
        self.assertNotIn(str(self.root), canonical(result))

    def test_forbidden_memory_paths_are_rejected_before_filesystem_access(self):
        for path in ("E:/unused.json", "e:\\unused.json", "/mnt/./e/unused.json", "C:/a/../b"):
            with self.subTest(path=path), patch("orchestrator.ranking_pipeline._guard_path") as guard:
                result = self.pipeline(snapshot=path).rank(request_fixture())
            guard.assert_not_called()
            self.assert_response(result, ["2", "3", "0", "1"], "llm_disabled")
            self.assertEqual(result["metadata"]["memory"]["fallback_reason"], "ValueError")

    def test_lookup_error_messages_are_not_exposed_to_llm_or_metadata(self):
        transport = self.transport()
        pipeline = self.pipeline(cfg=config_fixture(), transport=transport)
        with patch.object(type(pipeline.memory_ranker.memory), "lookup",
                          side_effect=ValueError("private-provider-body")):
            result = pipeline.rank(request_fixture())
        self.assertEqual(result["metadata"]["memory"]["fallback_reason"], "ValueError")
        self.assertNotIn("private-provider-body", canonical(result))
        self.assertNotIn("private-provider-body", canonical(transport.call_args.args[2]))

    def test_input_and_context_copies_hashes_and_frozen_snapshot_are_preserved(self):
        before = self.snapshot.read_bytes()
        original_context = deepcopy(self.context)
        pipeline = self.pipeline()
        self.context["seed"] = 100
        offered = request_fixture()
        saved = deepcopy(offered)
        result = pipeline.rank(offered)
        self.assertEqual(offered, saved)
        self.assertEqual(result["metadata"]["input_envelope_sha256"], digest(saved))
        enriched = deepcopy(saved)
        enriched["features"]["memory"] = result["metadata"]["memory"]
        enriched["features"]["deterministic_fallback_order"] = result["candidate_ids"]
        enriched["timeout_seconds"] = 30.0
        self.assertEqual(result["metadata"]["llm_enriched_envelope_sha256"], digest(enriched))
        self.assertEqual(result["metadata"]["memory"]["context_digest"], digest(original_context))
        self.assertEqual(result["metadata"]["memory"]["snapshot_sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(self.snapshot.read_bytes(), before)

    def test_timeout_leaves_worker_response_headroom(self):
        for original, expected in ((45, 30.0), (30, 25.0), (6, 1.0), (5, 2.5), (0.1, 0.05)):
            with self.subTest(timeout=original):
                transport = self.transport()
                result = self.pipeline(cfg=config_fixture(), transport=transport).rank(
                    request_fixture(timeout=original))
                self.assertEqual(transport.call_args.args[3], expected)
                self.assertEqual(result["metadata"]["worker_timeout_seconds"], original)
                self.assertEqual(result["metadata"]["llm_timeout_seconds"], expected)
                self.assertLess(expected, original)

    def test_repeated_request_and_reopen_reuse_ledger_without_new_call(self):
        transport = self.transport()
        ledger = self.root / "shared.sqlite"
        first = self.pipeline(cfg=config_fixture(), transport=transport, ledger=ledger)
        offered = request_fixture()
        original = first.rank(offered)
        repeated = first.rank(offered)
        reopened = self.pipeline(cfg=config_fixture(), transport=transport, ledger=ledger).rank(offered)
        self.assertEqual(transport.call_count, 1)
        for result in (repeated, reopened):
            self.assertEqual(result["candidate_ids"], original["candidate_ids"])
            self.assertTrue(result["metadata"]["cached"])
            self.assertFalse(result["metadata"]["llm_live"])
            self.assertFalse(result["metadata"]["attempted"])

    def test_invalid_request_never_queries_evidence_or_transport(self):
        transport = self.transport()
        pipeline = self.pipeline(cfg=config_fixture(), transport=transport)
        offered = request_fixture(candidates=["2", "2"])
        with patch.object(pipeline.memory_ranker, "apply") as apply:
            result = pipeline.rank(offered)
        apply.assert_not_called()
        transport.assert_not_called()
        self.assertEqual(result["fallback_reason"], "invalid_request")
        self.assertEqual(result["metadata"]["llm_fallback_reason"], "invalid_request")

    def test_unknown_modes_and_unregistered_rules_are_configuration_errors(self):
        with self.assertRaises(ValueError):
            self.pipeline(mode="sometimes")
        with self.assertRaises(ValueError):
            self.pipeline(base="generated-code")


if __name__ == "__main__":
    unittest.main()
