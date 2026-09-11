"""
Author: Perijn
Summary: Verifies translation of Halogen API ledger lines into llama-swap timings.
Usage: Run `python -m unittest docker/llama-swap-halogen/tests/test_halogen_telemetry_proxy.py`.
"""

import importlib.util
import json
import pathlib
import sys
import unittest

MODULE_PATH = (
    pathlib.Path(__file__).parents[1] / "scripts" / "halogen-telemetry-proxy.py"
)
SPEC = importlib.util.spec_from_file_location("halogen_telemetry_proxy", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class HalogenTelemetryProxyTests(unittest.TestCase):
    """Test ledger parsing and OpenAI response decoration."""

    def test_ledger_line_produces_llama_cpp_timings(self) -> None:
        """Translate the documented Halogen ledger schema exactly."""
        ledger = MODULE.Ledger()
        ledger.add_line(
            "serve_api: mtp 256 tok in 6.1s = 41.97 t/s | 92 rounds, commit 2.78/round | prompt 32768 (120 cached), prefill 23.0s\n"
        )

        timing = ledger.take(256, 32768, timeout=0)

        self.assertIsNotNone(timing)
        assert timing is not None
        self.assertEqual(timing.as_timings()["prompt_n"], 32768)
        self.assertEqual(timing.as_timings()["cache_n"], 120)
        self.assertEqual(timing.as_timings()["predicted_n"], 256)
        self.assertEqual(timing.as_timings()["prompt_ms"], 23000)
        self.assertEqual(timing.as_timings()["predicted_ms"], 6100)
        self.assertEqual(timing.as_timings()["predicted_per_second"], 41.97)

    def test_ledger_boundary_ignores_a_stale_identical_request(self) -> None:
        """Do not match an earlier equal-count ledger record after a new request starts."""
        ledger = MODULE.Ledger()
        ledger.add_line(
            "serve_api: mtp 2 tok in 1.0s = 2.0 t/s | 1 rounds, commit 1.0/round | prompt 8, prefill 1.0s\n"
        )
        marker = ledger.mark()
        ledger.add_line(
            "serve_api: mtp 2 tok in 3.0s = 0.67 t/s | 1 rounds, commit 1.0/round | prompt 8, prefill 2.0s\n"
        )

        timing = ledger.take(2, 8, timeout=0, after=marker)

        self.assertIsNotNone(timing)
        assert timing is not None
        self.assertEqual(3.0, timing.decode_seconds)

    def test_non_streaming_reply_receives_timings(self) -> None:
        """Preserve existing OpenAI response content while adding telemetry."""
        payload = (
            b'{"id":"chatcmpl-1","usage":{"prompt_tokens":12,"completion_tokens":3}}'
        )
        timing = MODULE.LedgerTiming(3, 0.1, 30.0, 12, 0, 0.02)

        decorated = json.loads(MODULE.add_timings(payload, timing))

        self.assertEqual(decorated["usage"]["completion_tokens"], 3)
        self.assertEqual(decorated["timings"]["prompt_per_second"], 600.0)
        self.assertEqual(decorated["timings"]["predicted_per_second"], 30.0)

    def test_timing_event_is_parseable_sse(self) -> None:
        """Emit a final SSE event that llama-swap's metrics parser can consume."""
        event = MODULE.timing_event(MODULE.LedgerTiming(1, 0.02, 50.0, 20, 5, 0.01))

        self.assertTrue(event.startswith(b"data: {"))
        self.assertTrue(event.endswith(b"\n\n"))
        self.assertEqual(json.loads(event[6:].strip())["timings"]["cache_n"], 5)


if __name__ == "__main__":
    unittest.main()
