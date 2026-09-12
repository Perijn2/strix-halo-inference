"""
Author: Perijn
Summary: Verifies that Ciru model-name normalization rewrites only the intended model field.
Usage: Run `python -m unittest docker/llama-swap-halogen/tests/test_ciru_model_proxy.py`.
"""

import http.client
import http.server
import importlib.util
import json
import pathlib
import socket
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer

MODULE_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "ciru-model-proxy.py"
SPEC = importlib.util.spec_from_file_location("ciru_model_proxy", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

CANONICAL = "ciru-halo-agent"
FULL_USAGE = {
    "prompt_tokens": 12,
    "completion_tokens": 5,
    "total_tokens": 17,
    "prompt_tokens_details": {"cached_tokens": 4},
}


class CiruModelProxyTests(unittest.TestCase):
    """Test alias parsing, request normalization, and response restoration."""

    def test_alias_list_drops_blank_duplicate_and_canonical_names(self) -> None:
        """Keep only usable aliases for the configured canonical name."""
        aliases = MODULE.parse_alias_list(
            "role/implementer, role/tester,,role/implementer,ciru-halo-agent", CANONICAL
        )

        self.assertEqual(aliases, ("role/implementer", "role/tester"))

    def test_body_model_reads_the_top_level_model_name(self) -> None:
        """Extract the routing name llama-swap forwards verbatim."""
        body = json.dumps({"model": "role/documenter", "max_tokens": 8}).encode()

        self.assertEqual(MODULE.body_model(body), "role/documenter")

    def test_body_model_ignores_non_json_and_non_object_payloads(self) -> None:
        """Never normalize a body whose shape carries no model name."""
        self.assertIsNone(MODULE.body_model(b""))
        self.assertIsNone(MODULE.body_model(b"not json"))
        self.assertIsNone(MODULE.body_model(b"[1, 2, 3]"))
        self.assertIsNone(MODULE.body_model(b'{"model": 42}'))

    def test_normalize_replaces_only_the_model_value(self) -> None:
        """Preserve every other byte of a large chat request."""
        body = (
            b'{"model":"role/implementer","messages":[{"role":"user",'
            b'"content":"port the parser"}],"temperature":0.6}'
        )

        normalized = MODULE.normalize_request(body, "role/implementer", CANONICAL)

        self.assertEqual(
            normalized,
            b'{"model":"ciru-halo-agent","messages":[{"role":"user",'
            b'"content":"port the parser"}],"temperature":0.6}',
        )

    def test_normalize_handles_spaced_json_and_keeps_other_fields(self) -> None:
        """Accept pretty-printed JSON from any client."""
        body = b'{\n  "model" : "role/tester",\n  "stream": true\n}'

        normalized = MODULE.normalize_request(body, "role/tester", CANONICAL)

        self.assertEqual(json.loads(normalized)["model"], CANONICAL)
        self.assertTrue(json.loads(normalized)["stream"])

    def test_escaped_model_text_inside_a_value_is_untouched(self) -> None:
        """Prompt text quoting a model field must not look like a field to rewrite."""
        body = json.dumps(
            {
                "model": CANONICAL,
                "messages": [
                    {"role": "user", "content": 'echo "model":"role/implementer" back'}
                ],
            }
        ).encode()

        pattern = MODULE.model_field_pattern("role/implementer")

        self.assertIsNone(pattern.search(body))
        self.assertEqual(MODULE.substitute_model(body, pattern, CANONICAL), body)

    def test_pattern_does_not_match_a_longer_similar_name(self) -> None:
        """Anchor on the closing quote so versioned names stay distinct."""
        pattern = MODULE.model_field_pattern(CANONICAL)

        self.assertIsNone(pattern.search(b'{"model":"ciru-halo-agent-v2"}'))

    def test_restore_response_maps_canonical_name_back_to_the_alias(self) -> None:
        """Return the caller's own ID so client-side assertions keep working."""
        line = (
            b'data: {"model":"ciru-halo-agent","choices":[{"delta":{"content":"hi"}}]}'
        )

        restored = MODULE.restore_response(
            line, MODULE.model_field_pattern(CANONICAL), "role/tester"
        )

        self.assertEqual(json.loads(restored[6:])["model"], "role/tester")
        self.assertIn(b'"content":"hi"', restored)

    def test_restore_response_leaves_other_payloads_untouched(self) -> None:
        """An error body without the canonical name passes through byte-exact."""
        payload = b'{"error":{"message":"engine busy"}}'

        restored = MODULE.restore_response(
            payload, MODULE.model_field_pattern(CANONICAL), "role/tester"
        )

        self.assertEqual(restored, payload)

    def test_round_trip_restores_the_original_alias(self) -> None:
        """A request alias survives the engine and comes back unchanged."""
        body = b'{"model":"role/documenter","max_tokens":4}'

        wire = MODULE.normalize_request(body, "role/documenter", CANONICAL)
        back = MODULE.restore_response(
            wire, MODULE.model_field_pattern(CANONICAL), "role/documenter"
        )

        self.assertEqual(back, body)

    def test_usage_counts_extracts_prompt_completion_and_cached_tokens(self) -> None:
        """Real vLLM usage blocks yield every timing anchor including cache hits."""
        payload = json.dumps({"usage": FULL_USAGE}).encode()

        self.assertEqual(MODULE.usage_counts(payload), (12, 5, 4))

    def test_usage_counts_tolerates_missing_or_malformed_payloads(self) -> None:
        """Nothing is invented when the response carries no usable counts."""
        self.assertEqual(MODULE.usage_counts(b"not json"), (None, None, 0))
        self.assertEqual(MODULE.usage_counts(b'{"usage": null}'), (None, None, 0))
        self.assertEqual(
            MODULE.usage_counts(b'{"usage": {"total_tokens": 7}}'), (None, None, 0)
        )

    def test_stream_timings_use_exact_wire_windows(self) -> None:
        """A stream's prefill and decode windows are measured, not guessed."""
        telemetry = MODULE.ChatTelemetry(started=100.0)
        telemetry.note_body(101.5)
        telemetry.note_body(111.5)
        telemetry.note_usage(50, 100, 20)

        timings = telemetry.as_timings(streaming=True)

        self.assertEqual(timings["prompt_n"], 50)
        self.assertEqual(timings["cache_n"], 20)
        self.assertEqual(timings["predicted_n"], 100)
        self.assertAlmostEqual(timings["prompt_ms"], 1500.0)
        self.assertAlmostEqual(timings["predicted_ms"], 10000.0)
        self.assertAlmostEqual(timings["prompt_per_second"], 20.0)
        self.assertAlmostEqual(timings["predicted_per_second"], 10.0)

    def test_non_stream_timings_report_conservative_end_to_end_windows(self) -> None:
        """An unsplittable reply reports the whole request in both windows."""
        telemetry = MODULE.ChatTelemetry(started=100.0)
        telemetry.note_body(105.0)
        telemetry.note_usage(10, 5, 0)

        timings = telemetry.as_timings(streaming=False)

        self.assertAlmostEqual(timings["prompt_ms"], 5000.0)
        self.assertAlmostEqual(timings["predicted_ms"], 5000.0)
        self.assertAlmostEqual(timings["prompt_per_second"], 2.0)
        self.assertAlmostEqual(timings["predicted_per_second"], 1.0)

    def test_timings_are_absent_without_usage_counts(self) -> None:
        """No timing record is fabricated from an unanchored measurement."""
        telemetry = MODULE.ChatTelemetry(started=100.0)
        telemetry.note_body(101.0)

        self.assertIsNone(telemetry.as_timings(streaming=True))
        self.assertIsNone(telemetry.as_timings(streaming=False))

    def test_collapsed_stream_decode_window_falls_back_to_the_whole_request(self) -> None:
        """A single-burst stream never divides by a zero decode window."""
        telemetry = MODULE.ChatTelemetry(started=100.0)
        telemetry.note_body(101.0)
        telemetry.note_usage(4, 1, 0)

        timings = telemetry.as_timings(streaming=True)

        self.assertAlmostEqual(timings["predicted_ms"], 1000.0)
        self.assertAlmostEqual(timings["predicted_per_second"], 1.0)

    def test_add_timings_decorates_objects_and_leaves_other_payloads(self) -> None:
        """Only JSON objects receive the timings block; everything else stays exact."""
        decorated = MODULE.add_timings(b'{"model": "x"}', {"prompt_n": 1})

        self.assertEqual(
            json.loads(decorated), {"model": "x", "timings": {"prompt_n": 1}}
        )
        self.assertEqual(MODULE.add_timings(b"nope", {"prompt_n": 1}), b"nope")
        self.assertEqual(MODULE.add_timings(b"[1]", {"prompt_n": 1}), b"[1]")

    def test_launcher_argv_binds_the_vendor_launcher_to_the_inner_port(self) -> None:
        """The proxy chooses which port serve.sh binds, keeping the managed port free."""
        self.assertEqual(
            MODULE.launcher_argv(MODULE.DEFAULT_ENTRYPOINT, 5801),
            [
                "bash",
                "/ornith/bundle/serve.sh",
                "--host",
                "127.0.0.1",
                "--port",
                "5801",
            ],
        )

    def test_launcher_argv_keeps_custom_entrypoint_arguments(self) -> None:
        """A custom launcher keeps its own flags ahead of the appended ones."""
        self.assertEqual(
            MODULE.launcher_argv("python /opt/serve.py --profile iu4", 6001),
            [
                "python",
                "/opt/serve.py",
                "--profile",
                "iu4",
                "--host",
                "127.0.0.1",
                "--port",
                "6001",
            ],
        )

    def test_inner_port_prefers_the_requested_port_when_free(self) -> None:
        """Keep the port-plus-one layout deterministic whenever nothing holds it."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            free_port = probe.getsockname()[1]

        self.assertEqual(MODULE.pick_inner_port(free_port), free_port)

    def test_inner_port_moves_when_the_requested_port_is_taken(self) -> None:
        """Never hand the engine a port another process already holds."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind(("127.0.0.1", 0))
            held.listen()
            taken_port = held.getsockname()[1]

            chosen_port = MODULE.pick_inner_port(taken_port)

        self.assertNotEqual(chosen_port, taken_port)


class FakeEngine(http.server.BaseHTTPRequestHandler):
    """Stand-in for the Ciru vLLM endpoint that serves exactly one model name."""

    seen: list[dict] = []
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802
        """Answer like vLLM: serve only the canonical name, 404 every other name."""
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        payload = json.loads(raw) if raw else {}
        self.seen.append(
            {
                "model": payload.get("model"),
                "normalized": self.headers.get("X-Ciru-Model-Normalized"),
                "declared_length": self.headers.get("Content-Length"),
                "received_bytes": len(raw),
                "body": raw,
            }
        )
        if payload.get("model") != CANONICAL:
            self._json(
                404,
                {
                    "error": {
                        "message": f"The model `{payload.get('model')}` does not exist.",
                        "type": "NotFoundError",
                        "code": 404,
                    }
                },
            )
            return
        usage = payload.get("usage")
        if payload.get("stream"):
            self._event_stream(usage if isinstance(usage, dict) else None)
            return
        self._json(
            200,
            {
                "id": "chatcmpl-1",
                "model": CANONICAL,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "ok"}}
                ],
                "usage": usage if isinstance(usage, dict) else {"total_tokens": 7},
            },
        )

    def do_GET(self) -> None:  # noqa: N802
        """Answer the readiness probe llama-swap polls."""
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _event_stream(self, usage: dict | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for text in ("he", "llo"):
            line = json.dumps(
                {
                    "id": "chatcmpl-1",
                    "model": CANONICAL,
                    "choices": [{"index": 0, "delta": {"content": text}}],
                }
            )
            self.wfile.write(f"data: {line}\n\n".encode("utf-8"))
            self.wfile.flush()
        if usage is not None:
            line = json.dumps(
                {
                    "id": "chatcmpl-1",
                    "model": CANONICAL,
                    "choices": [],
                    "usage": usage,
                }
            )
            self.wfile.write(f"data: {line}\n\n".encode("utf-8"))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, _format: str, *_args: object) -> None:
        """Keep test output quiet."""


class CiruModelProxyIntegrationTests(unittest.TestCase):
    """Drive the real handler against a fake engine to prove both rewrite directions."""

    def setUp(self) -> None:
        """Start a fake engine and a proxy listener wired to it."""
        FakeEngine.seen = []
        MODULE.CiruProxyHandler.warned.clear()
        MODULE.CiruProxyHandler.canonical_model = CANONICAL
        MODULE.CiruProxyHandler.canonical_pattern = MODULE.model_field_pattern(
            CANONICAL
        )
        MODULE.CiruProxyHandler.aliases = (
            "role/implementer",
            "role/tester",
            "role/documenter",
            "ornith-1.5-35b-a3b",
        )
        self.engine = ThreadingHTTPServer(("127.0.0.1", 0), FakeEngine)
        threading.Thread(target=self.engine.serve_forever, daemon=True).start()
        MODULE.CiruProxyHandler.inner_port = self.engine.server_address[1]
        self.proxy = ThreadingHTTPServer(("127.0.0.1", 0), MODULE.CiruProxyHandler)
        threading.Thread(target=self.proxy.serve_forever, daemon=True).start()

    def tearDown(self) -> None:
        """Stop both listeners."""
        for server in (self.proxy, self.engine):
            server.shutdown()
            server.server_close()

    def _post(self, payload: dict) -> tuple[int, bytes]:
        client = http.client.HTTPConnection(
            "127.0.0.1", self.proxy.server_address[1], timeout=10
        )
        try:
            client.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
            response = client.getresponse()
            return response.status, response.read()
        finally:
            client.close()

    def _post_with_headers(self, payload: dict) -> tuple[int, bytes, dict[str, str]]:
        client = http.client.HTTPConnection(
            "127.0.0.1", self.proxy.server_address[1], timeout=10
        )
        try:
            client.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
            response = client.getresponse()
            return response.status, response.read(), dict(response.getheaders())
        finally:
            client.close()

    def _stream(self, payload: dict) -> list[str]:
        client = http.client.HTTPConnection(
            "127.0.0.1", self.proxy.server_address[1], timeout=10
        )
        try:
            client.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
            text = client.getresponse().read().decode("utf-8")
            return [line for line in text.splitlines() if line.startswith("data: ")]
        finally:
            client.close()

    def test_alias_reaches_the_engine_as_the_canonical_name(self) -> None:
        """The engine only ever sees the name its release serves."""
        status, body = self._post(
            {
                "model": "role/implementer",
                "messages": [{"role": "user", "content": "hi"}],
            }
        )

        self.assertEqual(status, 200)
        self.assertEqual(FakeEngine.seen[0]["model"], CANONICAL)
        self.assertEqual(FakeEngine.seen[0]["normalized"], "role/implementer")
        self.assertIn(b'"content": "ok"', body)

    def test_rewritten_request_declares_its_new_length(self) -> None:
        """A rewritten model name changes the body size, so Content-Length must follow."""
        for alias in ("role/implementer", "role/tester", "ornith-1.5-35b-a3b"):
            with self.subTest(alias=alias):
                self._post({"model": alias, "messages": []})
                seen = FakeEngine.seen[-1]

                self.assertEqual(seen["model"], CANONICAL)
                self.assertEqual(int(seen["declared_length"]), seen["received_bytes"])
                self.assertEqual(json.loads(seen["body"])["model"], CANONICAL)

    def test_rewritten_response_declares_its_new_length(self) -> None:
        """The restored alias changes the reply size, so framing must be recomputed."""
        client = http.client.HTTPConnection(
            "127.0.0.1", self.proxy.server_address[1], timeout=10
        )
        client.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps({"model": "role/documenter", "messages": []}),
            headers={"Content-Type": "application/json"},
        )
        response = client.getresponse()
        body = response.read()
        declared = response.getheader("Content-Length")
        client.close()

        self.assertEqual(response.status, 200)
        self.assertEqual(int(declared), len(body))
        self.assertEqual(json.loads(body)["model"], "role/documenter")

    def test_chunked_request_is_rejected_with_a_clear_message(self) -> None:
        """A body this proxy cannot re-frame is refused instead of silently forwarded."""
        client = http.client.HTTPConnection(
            "127.0.0.1", self.proxy.server_address[1], timeout=10
        )
        payload = json.dumps({"model": "role/tester", "messages": []}).encode("utf-8")
        client.putrequest("POST", "/v1/chat/completions")
        client.putheader("Content-Type", "application/json")
        client.putheader("Transfer-Encoding", "chunked")
        client.endheaders()
        client.send(b"%x\r\n%s\r\n" % (len(payload), payload))
        client.send(b"0\r\n\r\n")
        response = client.getresponse()
        body = response.read()
        client.close()

        self.assertEqual(response.status, 400)
        self.assertIn("Content-Length", json.loads(body)["error"]["message"])
        self.assertEqual(FakeEngine.seen, [])

    def test_response_restores_the_caller_own_model_id(self) -> None:
        """The client sees the ID it asked for, not the vendor's internal name."""
        _, body = self._post({"model": "role/tester", "messages": []})

        self.assertEqual(json.loads(body)["model"], "role/tester")

    def test_streamed_response_restores_the_alias_on_every_event(self) -> None:
        """SSE deltas keep the caller's ID and the terminal event stays intact."""
        client = http.client.HTTPConnection(
            "127.0.0.1", self.proxy.server_address[1], timeout=10
        )
        client.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(
                {"model": "role/documenter", "stream": True, "messages": []}
            ),
            headers={"Content-Type": "application/json"},
        )
        response = client.getresponse()
        text = response.read().decode("utf-8")
        client.close()

        self.assertEqual(response.getheader("Content-Type"), "text/event-stream")
        events = [line for line in text.splitlines() if line.startswith("data: ")]
        self.assertEqual(len(events), 3)
        for event in events[:-1]:
            self.assertEqual(
                json.loads(event[len("data: ") :])["model"], "role/documenter"
            )
        self.assertEqual(events[-1], "data: [DONE]")
        self.assertIn(
            "hello",
            "".join(
                json.loads(event[len("data: ") :])["choices"][0]["delta"].get(
                    "content", ""
                )
                for event in events[:-1]
            ),
        )

    def test_unmapped_name_is_forwarded_unchanged_and_keeps_the_404(self) -> None:
        """An unknown ID is not silently rewritten; the engine's 404 surfaces as-is."""
        status, body = self._post({"model": "role/explorer", "messages": []})

        self.assertEqual(status, 404)
        self.assertEqual(FakeEngine.seen[0]["model"], "role/explorer")
        self.assertIsNone(FakeEngine.seen[0]["normalized"])
        self.assertIn("does not exist", json.loads(body)["error"]["message"])

    def test_canonical_name_passes_through_without_rewriting(self) -> None:
        """Calling the vendor name directly still works and adds no rewrite header."""
        status, body = self._post({"model": CANONICAL, "messages": []})

        self.assertEqual(status, 200)
        self.assertIsNone(FakeEngine.seen[0]["normalized"])
        self.assertEqual(json.loads(body)["model"], CANONICAL)

    def test_get_health_probe_passes_through(self) -> None:
        """llama-swap's readiness check reaches the engine untouched."""
        client = http.client.HTTPConnection(
            "127.0.0.1", self.proxy.server_address[1], timeout=10
        )
        client.request("GET", "/health")
        response = client.getresponse()
        payload = response.read()
        client.close()

        self.assertEqual(response.status, 200)
        self.assertEqual(payload, b"ok")
        self.assertEqual(FakeEngine.seen, [])

    def test_engine_down_yields_a_clear_502(self) -> None:
        """A dead upstream reports a gateway error instead of hanging the caller."""
        self.engine.shutdown()
        self.engine.server_close()

        status, body = self._post({"model": "role/implementer", "messages": []})

        self.assertEqual(status, 502)
        self.assertEqual(json.loads(body)["error"]["type"], "bad_gateway")

    def test_non_stream_chat_gets_wire_timings_from_usage_counts(self) -> None:
        """Real usage counts anchor a llama.cpp-compatible timings block for Activity."""
        status, body, headers = self._post_with_headers(
            {
                "model": "role/implementer",
                "messages": [{"role": "user", "content": "hi"}],
                "usage": FULL_USAGE,
            }
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers.get("X-Ciru-Telemetry"), "wire")
        reply = json.loads(body)
        self.assertEqual(reply["model"], "role/implementer")
        timings = reply["timings"]
        self.assertEqual(timings["prompt_n"], 12)
        self.assertEqual(timings["cache_n"], 4)
        self.assertEqual(timings["predicted_n"], 5)
        self.assertGreater(timings["prompt_ms"], 0.0)
        self.assertEqual(timings["prompt_ms"], timings["predicted_ms"])
        self.assertGreater(timings["predicted_per_second"], 0.0)

    def test_non_stream_without_real_counts_gets_no_fabricated_timings(self) -> None:
        """A usage block with only total_tokens anchors nothing and adds nothing."""
        _, body, _ = self._post_with_headers(
            {"model": "role/documenter", "messages": []}
        )

        self.assertNotIn("timings", json.loads(body))

    def test_streamed_chat_inserts_timing_event_before_done(self) -> None:
        """The derived timing event lands between the usage event and [DONE]."""
        events = self._stream(
            {
                "model": "role/tester",
                "stream": True,
                "messages": [],
                "usage": FULL_USAGE,
            }
        )

        self.assertEqual(events[-1], "data: [DONE]")
        self.assertEqual(list(json.loads(events[-2][6:])), ["timings"])
        timings = json.loads(events[-2][6:])["timings"]
        self.assertEqual(timings["prompt_n"], 12)
        self.assertEqual(timings["cache_n"], 4)
        self.assertEqual(timings["predicted_n"], 5)
        usage_event = json.loads(events[-3][6:])
        self.assertEqual(usage_event["model"], "role/tester")
        for event in events[:2]:
            self.assertEqual(json.loads(event[6:])["model"], "role/tester")

    def test_stream_without_usage_counts_receives_no_timing_event(self) -> None:
        """A client that never supplied usage gets its stream untouched."""
        events = self._stream(
            {"model": "role/documenter", "stream": True, "messages": []}
        )

        self.assertEqual(len(events), 3)
        self.assertEqual(events[-1], "data: [DONE]")
        for event in events[:-1]:
            self.assertNotIn("timings", json.loads(event[6:]))

    def test_canonical_direct_stream_still_receives_telemetry(self) -> None:
        """Telemetry keys off the chat path, not the alias rewrite path."""
        events = self._stream(
            {"model": CANONICAL, "stream": True, "messages": [], "usage": FULL_USAGE}
        )

        self.assertEqual(events[-1], "data: [DONE]")
        self.assertEqual(json.loads(events[-2][6:])["timings"]["predicted_n"], 5)
        self.assertEqual(json.loads(events[-3][6:])["model"], CANONICAL)


if __name__ == "__main__":
    unittest.main()
