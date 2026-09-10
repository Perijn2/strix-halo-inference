#!/usr/bin/env python3
"""
Author: Perijn
Summary: Proxies Halogen while adding its exact API-ledger timings to OpenAI replies.
Usage: halogen-launch invokes this program with Halogen's entrypoint and its
managed llama-swap port. It starts Halogen on the next loopback port and
listens on the managed port, preserving streaming responses while appending
llama.cpp-compatible timing fields for llama-swap activity metrics.
"""

from __future__ import annotations

import argparse
import collections
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

LEDGER = re.compile(
    r"serve_api: \w+ (\d+) tok in ([\d.]+)s = ([\d.]+) t/s \| "
    r"\d+ rounds, commit [\d.]+/round \| prompt (\d+)"
    r"(?: \((\d+) cached\))?, prefill ([\d.]+)s"
)
HOP_BY_HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"}


@dataclass(frozen=True)
class LedgerTiming:
    """Hold one exact engine decode and prefill measurement."""

    output_tokens: int
    decode_seconds: float
    tokens_per_second: float
    prompt_tokens: int
    cached_tokens: int
    prefill_seconds: float

    def as_timings(self) -> dict[str, float | int]:
        """Return the llama.cpp timing schema recognized by llama-swap."""
        return {
            "prompt_n": self.prompt_tokens,
            "cache_n": self.cached_tokens,
            "predicted_n": self.output_tokens,
            "prompt_ms": self.prefill_seconds * 1000,
            "predicted_ms": self.decode_seconds * 1000,
            "prompt_per_second": (self.prompt_tokens - self.cached_tokens) / self.prefill_seconds if self.prefill_seconds else 0,
            "predicted_per_second": self.tokens_per_second,
        }


class Ledger:
    """Collect Halogen's per-request API ledger and match completed replies."""

    def __init__(self) -> None:
        """Initialize an empty, synchronized bounded ledger queue."""
        self._items: collections.deque[LedgerTiming] = collections.deque(maxlen=256)
        self._condition = threading.Condition()

    def add_line(self, line: str) -> None:
        """Parse and retain one Halogen API output line when it contains timing."""
        match = LEDGER.search(line)
        if not match:
            return
        output, decode, tps, prompt, cached, prefill = match.groups()
        timing = LedgerTiming(int(output), float(decode), float(tps), int(prompt), int(cached or 0), float(prefill))
        with self._condition:
            self._items.append(timing)
            self._condition.notify_all()

    def take(self, output_tokens: int | None, prompt_tokens: int | None, timeout: float = 2.0) -> LedgerTiming | None:
        """Take the matching ledger record, waiting briefly for the API log flush."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                for index, timing in enumerate(self._items):
                    if output_tokens is not None and timing.output_tokens != output_tokens:
                        continue
                    if prompt_tokens is not None and timing.prompt_tokens != prompt_tokens:
                        continue
                    del self._items[index]
                    return timing
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)


def usage_counts(payload: bytes) -> tuple[int | None, int | None]:
    """Read OpenAI prompt and completion token counts from one JSON payload."""
    try:
        usage = json.loads(payload).get("usage", {})
        return usage.get("completion_tokens"), usage.get("prompt_tokens")
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
        return None, None


def timing_event(timing: LedgerTiming) -> bytes:
    """Encode a harmless final OpenAI-compatible SSE chunk with timing metadata."""
    return b"data: " + json.dumps({"timings": timing.as_timings()}, separators=(",", ":")).encode() + b"\n\n"


class HalogenProxyHandler(BaseHTTPRequestHandler):
    """Forward HTTP traffic to Halogen and decorate only chat-completion metrics."""

    protocol_version = "HTTP/1.1"
    ledger: ClassVar[Ledger]
    upstream_port: ClassVar[int]

    def do_GET(self) -> None:  # noqa: N802
        """Proxy a GET request unchanged."""
        self._proxy()

    def do_POST(self) -> None:  # noqa: N802
        """Proxy a POST request and add timing telemetry when applicable."""
        self._proxy()

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Proxy an OPTIONS request unchanged."""
        self._proxy()

    def _proxy(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        connection = http.client.HTTPConnection("127.0.0.1", self.upstream_port, timeout=3700)
        headers = {key: value for key, value in self.headers.items() if key.lower() not in HOP_BY_HOP_HEADERS | {"host"}}
        headers["Host"] = f"127.0.0.1:{self.upstream_port}"
        try:
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            content_type = response.getheader("Content-Type", "")
            is_chat = self.command == "POST" and self.path.split("?", 1)[0] == "/v1/chat/completions"
            is_stream = "text/event-stream" in content_type
            forwarded = {key: value for key, value in response.getheaders() if key.lower() not in HOP_BY_HOP_HEADERS | {"content-length"}}
            self.send_response(response.status, response.reason)
            for key, value in forwarded.items():
                self.send_header(key, value)
            self.send_header("Connection", "close")
            if is_chat:
                self.send_header("X-Halogen-Telemetry", "ledger")
            self.end_headers()
            if is_stream and is_chat:
                self._stream_with_timings(response)
            else:
                payload = response.read()
                if is_chat and response.status == 200:
                    timing = self.ledger.take(*usage_counts(payload))
                    if timing:
                        payload = add_timings(payload, timing)
                self.wfile.write(payload)
                self.wfile.flush()
        except (ConnectionError, OSError, http.client.HTTPException) as exc:
            self.send_error(502, f"Halogen upstream unavailable: {exc}")
        finally:
            connection.close()
            self.close_connection = True

    def _stream_with_timings(self, response: http.client.HTTPResponse) -> None:
        """Forward SSE immediately and insert timing event immediately before DONE."""
        buffered = b""
        output_tokens = prompt_tokens = None
        sent_timing = False
        while chunk := response.read(4096):
            buffered += chunk
            while b"\n" in buffered:
                line, buffered = buffered.split(b"\n", 1)
                stripped = line.strip()
                if stripped.startswith(b"data:") and stripped[5:].strip() != b"[DONE]":
                    output, prompt = usage_counts(stripped[5:].strip())
                    output_tokens = output if output is not None else output_tokens
                    prompt_tokens = prompt if prompt is not None else prompt_tokens
                if stripped == b"data: [DONE]" and not sent_timing:
                    timing = self.ledger.take(output_tokens, prompt_tokens)
                    if timing:
                        self.wfile.write(timing_event(timing))
                    sent_timing = True
                self.wfile.write(line + b"\n")
                self.wfile.flush()
        if buffered:
            self.wfile.write(buffered)
            self.wfile.flush()

    def log_message(self, _format: str, *_args: object) -> None:
        """Keep request logging in llama-swap rather than duplicating it here."""


def add_timings(payload: bytes, timing: LedgerTiming) -> bytes:
    """Attach a llama.cpp-compatible timings object to a JSON response."""
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return payload
    if not isinstance(decoded, dict):
        return payload
    decoded["timings"] = timing.as_timings()
    return json.dumps(decoded, separators=(",", ":")).encode()


def relay_output(process: subprocess.Popen[str], ledger: Ledger) -> None:
    """Mirror Halogen logs and collect its timing ledger until the child exits."""
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        ledger.add_line(line)


def main() -> int:
    """Start Halogen on an inner port and serve the telemetry-decorating proxy."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    if args.port >= 65535:
        parser.error("the managed port must leave one inner port available")

    env = os.environ.copy()
    env["HALOGEN_API_PORT"] = str(args.port + 1)
    child = subprocess.Popen([args.entrypoint, "all"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    ledger = Ledger()
    threading.Thread(target=relay_output, args=(child, ledger), daemon=True).start()
    HalogenProxyHandler.ledger = ledger
    HalogenProxyHandler.upstream_port = args.port + 1
    server = ThreadingHTTPServer(("127.0.0.1", args.port), HalogenProxyHandler)

    def stop(_signum: int, _frame: object) -> None:
        """Stop proxy traffic and forward llama-swap's termination signal."""
        server.shutdown()
        if child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if child.poll() is None:
            child.terminate()
    return child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
