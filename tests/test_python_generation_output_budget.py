"""Injected transport only: a larger response must reserve its full cost."""
from decimal import Decimal
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from orchestrator.python_skill_discovery import PythonSkillDiscoveryService
from tests.test_skill_discovery import config, synthesis

class PythonOutputBudgetTests(unittest.TestCase):
    def test_explicit_larger_output_is_priced_and_budget_rejects_before_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(max_output_tokens=1024)
            service = PythonSkillDiscoveryService(cfg, Path(directory) / "preview.sqlite")
            packet = synthesis()
            small = service.preview(packet)
            packet["context"]["python_output_tokens"] = 2048
            large = service.preview(packet)
            self.assertEqual(service.config["max_output_tokens"], 1024)
            self.assertEqual(large["body"]["max_output_tokens"], 2048)
            self.assertEqual(large["cost_estimate"]["output_tokens_cap"], 2048)
            self.assertGreater(Decimal(large["cost_estimate"]["usd"]),
                               Decimal(small["cost_estimate"]["usd"]))
            limit = (Decimal(large["cost_estimate"]["usd"]) + Decimal(small["cost_estimate"]["usd"])) / 2
            transport = Mock()
            guarded = PythonSkillDiscoveryService(dict(cfg, max_usd=str(limit)),
                Path(directory) / "guard.sqlite", transport=transport)
            with patch.dict(os.environ, {"DISCOVERY_FIXTURE_KEY": "test-not-real"}):
                result = guarded.propose(packet)
            self.assertEqual(result["fallback_reason"], "cost_budget_exhausted")
            transport.assert_not_called()

    def test_invalid_output_ceiling_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            service = PythonSkillDiscoveryService(config(), Path(directory) / "unused.sqlite")
            for value in (0, True, 4097, "2048"):
                packet = synthesis()
                packet["context"]["python_output_tokens"] = value
                with self.assertRaisesRegex(ValueError, "invalid_python_request_output_cap"):
                    service.preview(packet)

if __name__ == "__main__":
    unittest.main()
