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
        if payload.get("stream"):
            self._event_stream()
            return
        self._json(
            200,
            {
                "id": "chatcmpl-1",
                "model": CANONICAL,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "ok"}}
                ],
                "usage": {"total_tokens": 7},
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

    def _event_stream(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
