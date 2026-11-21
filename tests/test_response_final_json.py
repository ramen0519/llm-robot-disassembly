"""API-free transport fixtures; malformed responses never become chosen actions."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from orchestrator.response_final_json import decode_final_json
from orchestrator.explicit_action_service import ExplicitActionService, _parse_output
from orchestrator.explicit_action_provider_v5 import encode_provider_decision_v5
from worker.path_probe_round import canonical_path_round_response
from tests.test_explicit_provider_v4 import round_request
from tests.test_explicit_action_service import provider, config


def phased(value):
    raw = provider(value)
    raw["output"][0]["phase"] = "final_answer"
    note = deepcopy(raw["output"][0])
    note.update(phase="commentary", content=[dict(type="output_text", text="PRIVATE_DIAGNOSTIC_MARKER")])
    raw["output"].insert(0, note)
    return raw


class FinalJSONTests(unittest.TestCase):
    def test_commentary_is_ignored_only_with_one_explicit_final(self):
        raw = phased({"decision": 1})
        before, diag = deepcopy(raw), {}
        self.assertEqual(decode_final_json(raw, diag), {"decision": 1})
        self.assertEqual(raw, before)
        self.assertEqual(diag["ignored_commentary_messages"], 1)
        self.assertNotIn("PRIVATE_DIAGNOSTIC_MARKER", json.dumps(diag))
        with self.assertRaisesRegex(ValueError, "invalid_response"):
            _parse_output(raw)  # legacy parser behavior retained

    def test_ambiguous_incomplete_unknown_and_refused_responses_fail_closed(self):
        variants = []
        raw = phased({}); raw["output"][-1].pop("phase"); variants.append((raw, "ambiguous_final_phase"))
        raw = phased({}); raw["output"].append(deepcopy(raw["output"][-1])); variants.append((raw, "ambiguous_final_phase"))
        raw = phased({}); raw["output"].reverse(); variants.append((raw, "ambiguous_final_phase"))
        raw = phased({}); raw["output"][0]["phase"] = "PRIVATE_DIAGNOSTIC_MARKER"; variants.append((raw, "unknown_message_phase"))
        raw = phased({}); raw["status"] = "incomplete"; variants.append((raw, "llm_incomplete"))
        raw = phased({}); raw["output"][0]["content"] = [dict(type="refusal", refusal="PRIVATE_DIAGNOSTIC_MARKER")]
        variants.append((raw, "llm_refusal"))
        raw = phased({}); raw["output"].insert(0, dict(type="function_call")); variants.append((raw, "unexpected_output_item"))
        for raw, code in variants:
            diag = {}
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, code):
                decode_final_json(raw, diag)
            self.assertEqual(diag["code"], code)
            self.assertNotIn("PRIVATE_DIAGNOSTIC_MARKER", json.dumps(diag))

    def test_json_error_diagnostics_never_store_keys_or_text(self):
        for text, code in [
            ('{"PRIVATE_DIAGNOSTIC_MARKER":1,"PRIVATE_DIAGNOSTIC_MARKER":2}', "duplicate_json_key"),
            ('{"PRIVATE_DIAGNOSTIC_MARKER":NaN}', "nonfinite_json_number"),
            ('{"PRIVATE_DIAGNOSTIC_MARKER":', "invalid_json"),
            ('[]', "final_json_not_object"),
        ]:
            raw, diag = provider({}), {}
            raw["output"][0]["content"][0]["text"] = text
            with self.assertRaisesRegex(ValueError, code):
                decode_final_json(raw, diag)
            self.assertNotIn("PRIVATE_DIAGNOSTIC_MARKER", json.dumps(diag))

    def test_v5_handles_phase_and_rejects_stale_id_without_retry(self):
        for stale in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                request = round_request()
                wire = encode_provider_decision_v5(request, canonical_path_round_response(request, None))
                if stale:
                    wire["request_id"] = "wrong"
                transport = Mock(return_value=phased(wire))
                service = ExplicitActionService(config(), Path(directory) / "budget.sqlite", transport,
                    provider_contract="explicit_action_provider_v5", fallback_policy="path_stage_v1")
                with patch.dict(os.environ, {"EXPLICIT_FIXTURE_KEY": "fixture-only"}):
                    result = service.handle(request)
                    cached = service.handle(request)
                self.assertEqual(transport.call_count, 1)
                self.assertTrue(cached["metadata"]["cached"])
                self.assertEqual(result["metadata"]["decision_validation_stage"],
                                 "decision_contract" if stale else "accepted")
                self.assertEqual(result["fallback_reason"], "invalid_action_response" if stale else None)
                self.assertEqual(result["metadata"]["provider_decode_diagnostics"]["code"], "decoded_object")
                self.assertNotIn("PRIVATE_DIAGNOSTIC_MARKER", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
