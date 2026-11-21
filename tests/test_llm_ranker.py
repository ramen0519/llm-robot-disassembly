"""Offline transport fixtures only: these tests never call an external API."""
from copy import deepcopy
from contextlib import closing
from decimal import Decimal
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orchestrator.llm_ranker import RankService, BASE_URL
from orchestrator.batch_budget import BatchPolicyConflict


KEY = "fixture-only-key-never-log-this"


def envelope(request_id="fixture:r1"):
    return {"schema_version": 1, "kind": "rank_request", "request": {
        "run_id": "fixture", "request_id": request_id, "state_id": "state0",
        "assembled_parts": ["7", "1", "5", "6"],
        "candidate_ids": ["7", "1", "5"], "base_part": "6"},
        "features": {"description": "순위 힌트일 뿐 물리 판정이 아님"}, "timeout_seconds": 4}


def config(**overrides):
    return {"enabled": True, "model": "fixture-model", "api_key_env": "RANKER_TEST_KEY",
            "max_calls": 10, "max_usd": "1", "input_usd_per_million": "2",
            "output_usd_per_million": "4", **overrides}


def provider_response(request, **changes):
    value = {"request_id": request["request_id"], "state_id": request["state_id"],
             "candidate_ids": list(reversed(request["candidate_ids"])), **changes}
    return {"id": "resp_fixture", "model": "fixture-model-actual", "status": "completed",
            "error": None, "output": [{"type": "message", "role": "assistant",
                "status": "completed", "content": [{"type": "output_text", "text": json.dumps(value)}]}],
            "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}}


class RankServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="asap_llm_ranker_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.serial = 0
        self.env = patch.dict(os.environ, {"RANKER_TEST_KEY": KEY})
        self.env.start()
        self.addCleanup(self.env.stop)
        # Even a regression in fixture injection cannot issue a real HTTP request.
        self.network = patch("urllib.request.OpenerDirector.open",
                             side_effect=AssertionError("network_forbidden_in_test"))
        self.network_mock = self.network.start()
        self.addCleanup(self.network.stop)
        self.addCleanup(self.network_mock.assert_not_called)

    def service(self, cfg=None, transport=None, path=None):
        self.serial += 1
        return RankService(config() if cfg is None else cfg,
                           path or self.root / ("ledger%d.sqlite" % self.serial), transport)

    def transport(self):
        return Mock(side_effect=lambda url, headers, body, timeout:
                    provider_response(json.loads(body["input"][0]["content"])["request"]))

    @staticmethod
    def records(service):
        with closing(sqlite3.connect(service.ledger_path)) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute("SELECT * FROM rank_requests")]

    def assert_fallback(self, response, reason, offered=None):
        self.assertEqual(response["fallback_reason"], reason)
        self.assertEqual(response["candidate_ids"], offered or ["7", "1", "5"])
        self.assertEqual(response["kind"], "rank_response")

    def test_default_and_zero_caps_never_call_transport(self):
        for cfg in ({}, {"enabled": True}, config(enabled=False), config(max_calls=0),
                    config(max_usd=0), config(max_usd="0")):
            with self.subTest(cfg=cfg):
                transport = self.transport()
                service = self.service(cfg, transport)
                result = service.rank(envelope())
                self.assert_fallback(result, "llm_disabled")
                self.assertFalse(result["metadata"]["attempted"])
                self.assertFalse(result["metadata"]["llm_live"])
                self.assertEqual(self.records(service)[0]["reserved_call"], 0)
                transport.assert_not_called()

    def test_default_http_transport_is_not_invoked_when_disabled(self):
        self.assert_fallback(self.service({}).rank(envelope()), "llm_disabled")

    def test_valid_permutation_schema_and_cost_evidence(self):
        transport = self.transport()
        service = self.service(transport=transport)
        offered = envelope()
        saved = deepcopy(offered)
        result = service.rank(offered)
        self.assertIsNone(result["fallback_reason"])
        self.assertEqual(result["candidate_ids"], ["5", "1", "7"])
        self.assertEqual(offered, saved)
        url, headers, body, timeout = transport.call_args.args
        self.assertEqual(url, BASE_URL + "/responses")
        self.assertEqual(headers["Authorization"], "Bearer " + KEY)
        self.assertEqual(timeout, 4)
        self.assertFalse(body["store"])
        schema = body["text"]["format"]
        self.assertTrue(schema["strict"])
        self.assertFalse(schema["schema"]["additionalProperties"])
        self.assertEqual(set(schema["schema"]["required"]), {"request_id", "state_id", "candidate_ids"})
        metadata = result["metadata"]
        self.assertTrue(metadata["attempted"])
        self.assertFalse(metadata["llm_live"])
        self.assertEqual(metadata["model_returned"], "fixture-model-actual")
        self.assertEqual(metadata["response_id"], "resp_fixture")
        self.assertEqual(metadata["usage"]["total_tokens"], 120)
        self.assertEqual(Decimal(metadata["actual_cost_from_usage_usd"]), Decimal("0.00028"))
        estimate = metadata["cost_estimate"]
        self.assertTrue(estimate["is_estimate"])
        self.assertFalse(estimate["billing_cap_guaranteed"])
        self.assertGreater(estimate["input_tokens_upper_estimate"], len(json.dumps(body).encode()))
        self.assertNotIn(KEY, service.ledger_path.read_bytes().decode("latin1"))

    def test_invalid_configuration_fails_closed(self):
        for changes in ({"model": None}, {"model": ""}, {"base_url": "https://example.invalid/v1"},
                        {"max_calls": -1}, {"max_calls": True}, {"max_usd": "NaN"},
                        {"input_usd_per_million": None}, {"output_usd_per_million": -1},
                        {"max_output_tokens": 100000}, {"reasoning_effort": "unlimited"},
                        {"request_overhead_tokens": 0}, {"api_key": "do-not-use-inline-keys"},
                        {"api_key_env": "bad\nname"}):
            with self.subTest(changes=changes):
                transport = self.transport()
                result = self.service(config(**changes), transport).rank(envelope())
                self.assert_fallback(result, "invalid_config")
                transport.assert_not_called()

    def test_missing_key_is_not_read_from_alternative_environment(self):
        transport = self.transport()
        with patch.dict(os.environ, {}, clear=True):
            result = self.service(transport=transport).rank(envelope())
        self.assert_fallback(result, "missing_api_key")
        transport.assert_not_called()

    def test_invalid_envelopes_do_not_reserve_or_call(self):
        changes = ({"timeout_seconds": 0}, {"timeout_seconds": float("nan")},
                   {"features": []}, {"schema_version": 2}, {"kind": "physics_verdict"})
        for change in changes:
            with self.subTest(change=change):
                transport = self.transport()
                service = self.service(transport=transport)
                self.assert_fallback(service.rank({**envelope(), **change}), "invalid_request")
                self.assertEqual(self.records(service), [])
                transport.assert_not_called()

    def test_request_candidates_must_be_unique_subset_excluding_base(self):
        for offered in (["1", "1"], ["not-offered"], ["6"], [1]):
            with self.subTest(offered=offered):
                request = envelope()
                request["request"]["candidate_ids"] = offered
                transport = self.transport()
                self.assertEqual(self.service(transport=transport).rank(request)["fallback_reason"],
                                 "invalid_request")
                transport.assert_not_called()

    def test_omitted_duplicate_foreign_and_extra_candidates_fallback(self):
        for candidates in (["7", "1"], ["7", "1", "1"], ["7", "1", "9"],
                           ["7", "1", "5", "6"], [7, "1", "5"]):
            with self.subTest(candidates=candidates):
                transport = Mock(return_value=provider_response(envelope()["request"],
                                                               candidate_ids=candidates))
                service = self.service(transport=transport)
                self.assert_fallback(service.rank(envelope()), "invalid_permutation")
                self.assertEqual(transport.call_count, 1)
                self.assertEqual(self.records(service)[0]["reserved_call"], 1)

    def test_stale_ids_and_extra_response_fields_fallback(self):
        for changes, reason in (({"request_id": "old"}, "stale_request_or_state"),
                                ({"state_id": "old"}, "stale_request_or_state"),
                                ({"physics_verdict": "PASS"}, "invalid_response")):
            with self.subTest(changes=changes):
                transport = Mock(return_value=provider_response(envelope()["request"], **changes))
                self.assert_fallback(self.service(transport=transport).rank(envelope()), reason)

    def test_refusal_incomplete_api_error_and_malformed_output_fallback(self):
        refusal = provider_response(envelope()["request"])
        refusal["output"][0]["content"] = [{"type": "refusal", "refusal": "private provider text"}]
        cases = [(refusal, "llm_refusal"), ({"status": "incomplete"}, "llm_incomplete"),
                 ({"status": "failed", "error": {"message": KEY}}, "api_error"),
                 ({"status": "completed", "output": []}, "invalid_response"),
                 ([], "invalid_response")]
        for raw, reason in cases:
            with self.subTest(reason=reason):
                service = self.service(transport=Mock(return_value=raw))
                result = service.rank(envelope())
                self.assert_fallback(result, reason)
                self.assertNotIn(KEY, json.dumps(result))

    def test_duplicate_json_keys_are_rejected(self):
        raw = provider_response(envelope()["request"])
        raw["output"][0]["content"][0]["text"] = (
            '{"candidate_ids":["5","1","7"],"request_id":"old",'
            '"request_id":"fixture:r1","state_id":"state0"}')
        self.assert_fallback(self.service(transport=Mock(return_value=raw)).rank(envelope()),
                             "invalid_response")

    def test_timeout_and_provider_exception_never_log_messages_or_retry(self):
        for error, reason in ((TimeoutError(KEY), "api_timeout"),
                              (urllib.error.URLError(TimeoutError(KEY)), "api_timeout"),
                              (urllib.error.HTTPError(BASE_URL, 429, KEY, {}, None), "api_error"),
                              (RuntimeError(KEY), "api_error")):
            with self.subTest(error=type(error).__name__):
                transport = Mock(side_effect=error)
                service = self.service(transport=transport)
                result = service.rank(envelope())
                self.assert_fallback(result, reason)
                self.assertEqual(transport.call_count, 1)
                self.assertNotIn(KEY, json.dumps(result))
                self.assertNotIn(KEY.encode(), service.ledger_path.read_bytes())
                self.assertEqual(result["metadata"]["error_type"], type(error).__name__)
                if isinstance(error, urllib.error.HTTPError):
                    self.assertEqual(result["metadata"]["status_code"], 429)
                again = service.rank(envelope())
                self.assert_fallback(again, reason)
                self.assertFalse(again["metadata"]["attempted"])
                self.assertEqual(transport.call_count, 1)

    def test_echoed_credentials_in_provider_metadata_are_not_saved(self):
        raw = provider_response(envelope()["request"])
        raw.update(model="prefix" + KEY, id=KEY)
        service = self.service(transport=Mock(return_value=raw))
        result = service.rank(envelope())
        self.assertIsNone(result["metadata"]["model_returned"])
        self.assertIsNone(result["metadata"]["response_id"])
        self.assertNotIn(KEY.encode(), service.ledger_path.read_bytes())

    def test_completed_request_reused_after_service_restart_without_call(self):
        transport = self.transport()
        service = self.service(transport=transport)
        first = service.rank(envelope())
        restarted = self.service(transport=transport, path=service.ledger_path)
        second = restarted.rank(envelope())
        self.assertEqual(first["candidate_ids"], second["candidate_ids"])
        self.assertTrue(second["metadata"]["cached"])
        self.assertFalse(second["metadata"]["attempted"])
        self.assertEqual(first["metadata"]["call_id"], second["metadata"]["call_id"])
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(len(self.records(service)), 1)

    def test_same_id_payload_feature_timeout_or_config_change_rejected(self):
        transport = self.transport()
        service = self.service(transport=transport)
        service.rank(envelope())
        for changed in ({**envelope(), "features": {"new": True}},
                        {**envelope(), "timeout_seconds": 5}):
            self.assert_fallback(service.rank(changed), "request_id_conflict")
        with self.assertRaises(BatchPolicyConflict):
            self.service(config(model="another-model"), transport, service.ledger_path)
        self.assertEqual(transport.call_count, 1)

    def test_disabled_requests_also_use_idempotency_ledger(self):
        service = self.service({})
        service.rank(envelope())
        changed = envelope()
        changed["request"]["state_id"] = "other"
        self.assert_fallback(service.rank(changed), "request_id_conflict")

    def test_cost_cap_is_checked_before_transport_and_reserves_nothing(self):
        transport = self.transport()
        service = self.service(config(max_usd="0.00000001"), transport)
        result = service.rank(envelope())
        self.assert_fallback(result, "cost_budget_exhausted")
        transport.assert_not_called()
        self.assertEqual(self.records(service)[0]["reserved_usd"], "0")

    def test_call_cap_is_global_across_request_ids_and_config_changes(self):
        transport = self.transport()
        service = self.service(config(max_calls=1), transport)
        service.rank(envelope())
        with self.assertRaises(BatchPolicyConflict):
            self.service(config(max_calls=1, model="another-model"), transport, service.ledger_path)
        self.assert_fallback(service.rank(envelope("fixture:r2")), "call_budget_exhausted")
        self.assertEqual(transport.call_count, 1)

    def test_failed_call_reservation_counts_toward_cumulative_cost(self):
        transport = Mock(side_effect=TimeoutError("fixture_timeout"))
        service = self.service(transport=transport)
        first = service.rank(envelope())
        limit = str(Decimal(first["metadata"]["cost_estimate"]["usd"]) * Decimal("1.5"))
        with self.assertRaises(BatchPolicyConflict):
            self.service(config(max_usd=limit), transport, service.ledger_path)
        self.assertEqual(transport.call_count, 1)

    def test_unexpected_usage_over_estimate_blocks_later_call(self):
        raw = provider_response(envelope()["request"])
        raw["usage"] = {"input_tokens": 1000000, "output_tokens": 20, "total_tokens": 1000020}
        transport = Mock(return_value=raw)
        service = self.service(config(max_usd="1"), transport)
        first = service.rank(envelope())
        self.assertTrue(first["metadata"]["usage_cost_exceeds_reservation"])
        self.assert_fallback(service.rank(envelope("fixture:r2")), "cost_budget_exhausted")
        self.assertEqual(transport.call_count, 1)

    def test_crash_after_committed_reservation_never_resends(self):
        class SimulatedProcessDeath(BaseException):
            pass
        transport = Mock(side_effect=SimulatedProcessDeath())
        service = self.service(transport=transport)
        with self.assertRaises(SimulatedProcessDeath):
            service.rank(envelope())
        row = self.records(service)[0]
        self.assertEqual(row["status"], "pending")
        self.assertEqual(row["reserved_call"], 1)
        restarted = self.service(transport=transport, path=service.ledger_path)
        self.assert_fallback(restarted.rank(envelope()), "unknown_outcome")
        self.assert_fallback(restarted.rank(envelope()), "unknown_outcome")
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(self.records(restarted)[0]["reserved_call"], 1)

    def test_result_write_failure_keeps_reservation_and_prevents_resend(self):
        transport = self.transport()
        service = self.service(transport=transport)
        reservation_context = service._connect()
        with patch.object(service, "_connect", side_effect=[
                reservation_context, sqlite3.OperationalError(KEY)]):
            result = service.rank(envelope())
        self.assert_fallback(result, "ledger_error")
        self.assertNotIn(KEY, json.dumps(result))
        self.assertEqual(self.records(service)[0]["status"], "pending")
        self.assert_fallback(service.rank(envelope()), "unknown_outcome")
        self.assertEqual(transport.call_count, 1)

    def test_reservation_is_visible_before_transport_is_called(self):
        service = None
        def transport(url, headers, body, timeout):
            row = self.records(service)[0]
            self.assertEqual(row["status"], "pending")
            self.assertEqual(row["reserved_call"], 1)
            return provider_response(envelope()["request"])
        service = self.service(transport=transport)
        self.assertIsNone(service.rank(envelope())["fallback_reason"])

    def test_config_and_transport_cannot_mutate_callers_snapshot(self):
        settings = config()
        service = self.service(settings, self.transport())
        settings.update(enabled=False, model="other")
        self.assertIsNone(service.rank(envelope())["fallback_reason"])

    def test_non_c_drive_is_rejected_lexically_before_filesystem_access(self):
        with patch("pathlib.Path.lstat", side_effect=AssertionError("must_not_touch_drive")):
            with self.assertRaisesRegex(ValueError, "ledger_drive_not_allowed"):
                RankService({}, Path("E:/never-open/ledger.sqlite"))


if __name__ == "__main__":
    unittest.main()
