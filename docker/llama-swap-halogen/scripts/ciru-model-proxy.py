#!/usr/bin/env python3
"""
Author: Perijn
Summary: Presents llama-swap's stable role IDs to a vLLM release that serves exactly one model name.
Usage: llama-swap runs `ciru-model-proxy --port ${PORT}`. The proxy starts the vendor launcher on the
       next free loopback port, rewrites the request body's top-level model field to the single name the
       release serves, and rewrites that name back to the caller's own ID in the response.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, ClassVar

DEFAULT_ENTRYPOINT = "bash /ornith/bundle/serve.sh"
DEFAULT_CANONICAL_MODEL = "ciru-halo-agent"
DEFAULT_ALIASES = (
    "role/implementer",
    "role/tester",
    "role/documenter",
    "ornith-1.5-35b-a3b",
)
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})
REQUEST_STRIP_HEADERS = HOP_BY_HOP_HEADERS | {"host", "expect", "content-length"}
RESPONSE_STRIP_HEADERS = HOP_BY_HOP_HEADERS | {"content-length"}
MAX_NORMALIZE_BYTES = 32 * 1024 * 1024
MAX_BODY_BYTES = 128 * 1024 * 1024
MAX_WARNED_NAMES = 128
STREAM_CHUNK_BYTES = 65536


def parse_alias_list(raw: str, canonical_model: str) -> tuple[str, ...]:
    """Parse a comma-separated alias list, dropping blanks and the canonical name itself."""
    names = [item.strip() for item in raw.split(",")]
    unique: list[str] = []
    for name in names:
        if not name or name == canonical_model or name in unique:
            continue
        unique.append(name)
    return tuple(unique)


def body_model(body: bytes) -> str | None:
    """Return the top-level model name carried by a JSON request body."""
    if not body:
        return None
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.get("model")
    return name if isinstance(name, str) else None


def model_field_pattern(name: str) -> re.Pattern[bytes]:
    """Match a model field whose value is exactly this name, in compact or spaced JSON."""
    return re.compile(rb'("model"\s*:\s*)"' + re.escape(name.encode("utf-8")) + rb'"')


def substitute_model(
    payload: bytes, pattern: re.Pattern[bytes], replacement: str
) -> bytes:
    """Swap one specific model value while leaving every other byte of the payload intact."""
    value = replacement.encode("utf-8")
    return pattern.sub(
        lambda match: match.group(1) + b'"' + value + b'"', payload, count=1
    )


def normalize_request(body: bytes, alias: str, canonical_model: str) -> bytes:
    """Rewrite one known alias to the canonical served name inside a request body."""
    return substitute_model(body, model_field_pattern(alias), canonical_model)


def restore_response(
    payload: bytes, canonical_pattern: re.Pattern[bytes], alias: str
) -> bytes:
    """Restore the caller's own model ID over the canonical name in one response payload."""
    return substitute_model(payload, canonical_pattern, alias)


class Engine:
    """Run the vendor launcher in its own process group so the whole engine tree dies on demand."""

    def __init__(self, argv: list[str]) -> None:
        """Start the launcher now; the proxy answers requests long before the engine is ready."""
        self.argv = argv
        self.stopping = False
        self.process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

    def stop(self, grace_seconds: float) -> None:
        """Terminate the whole process group and escalate to a kill when it will not exit."""
        self.stopping = True
        try:
            group = os.getpgid(self.process.pid)
        except ProcessLookupError:
            return
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            self.process.wait(grace_seconds)
            return
        except subprocess.TimeoutExpired:
            pass
        print(
            f"ciru-model-proxy: engine ignored SIGTERM within {grace_seconds:.0f}s, "
            "sending SIGKILL",
            file=sys.stderr,
            flush=True,
        )
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            return
        try:
            self.process.wait(10)
        except subprocess.TimeoutExpired:
            print(
                "ciru-model-proxy: engine process group survived SIGKILL",
                file=sys.stderr,
                flush=True,
            )

    def watch(self, on_exit: Callable[[int], None]) -> None:
        """Notify the supervisor once when the engine exits by itself."""
        code = self.process.wait()
        if not self.stopping:
            on_exit(code if code is not None else 1)


def launcher_argv(entrypoint: str, inner_port: int) -> list[str]:
    """Build the vendor launcher command line bound to the inner loopback port."""
    return shlex.split(entrypoint) + ["--host", "127.0.0.1", "--port", str(inner_port)]


def pick_inner_port(preferred: int) -> int:
    """Use the preferred loopback port when free, otherwise accept an OS-assigned one."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", preferred))
            return preferred
    except OSError:
        pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class CiruProxyHandler(BaseHTTPRequestHandler):
    """Forward traffic to the Ciru engine and keep model naming consistent both ways."""

    protocol_version = "HTTP/1.1"
    inner_port: ClassVar[int]
    canonical_model: ClassVar[str]
    canonical_pattern: ClassVar[re.Pattern[bytes]]
    aliases: ClassVar[tuple[str, ...]]
    read_timeout: ClassVar[float] = 3600.0
    warned: ClassVar[set[str]] = set()
    warned_lock: ClassVar[threading.Lock] = threading.Lock()

    def do_GET(self) -> None:  # noqa: N802
        """Proxy a GET request unchanged."""
        self._proxy()

    def do_POST(self) -> None:  # noqa: N802
        """Proxy a POST request and normalize its model field when it carries a known alias."""
        self._proxy()

    def do_PUT(self) -> None:  # noqa: N802
        """Proxy a PUT request with the same model normalization rules."""
        self._proxy()

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Proxy an OPTIONS request unchanged."""
        self._proxy()

    def _alias_to_normalize(self, body: bytes) -> str | None:
        """Return the alias to rewrite, or None when this body must pass through untouched."""
        if self.command not in BODY_METHODS or not body:
            return None
        content_type = (self.headers.get("Content-Type") or "").lower()
        if content_type and "json" not in content_type:
            return None
        if len(body) > MAX_NORMALIZE_BYTES:
            self._warn(
                f"request body of {len(body)} bytes exceeds the "
                f"{MAX_NORMALIZE_BYTES}-byte normalization cap"
            )
            return None
        name = body_model(body)
        if name is None:
            return None
        if name in self.aliases:
            return name
        if name != self.canonical_model:
            self._warn(f"unmapped model name {name!r} passed through unchanged")
        return None

    def _read_request_body(self) -> tuple[bytes, str | None]:
        """Read the request body, or return the reason it cannot be forwarded."""
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            return b"", (
                "chunked request bodies are not supported by the Ciru model proxy; "
                "send a Content-Length header"
            )
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return b"", "invalid Content-Length header"
        if length < 0:
            return b"", "negative Content-Length header"
        if length > MAX_BODY_BYTES:
            return b"", f"request body exceeds the {MAX_BODY_BYTES}-byte proxy cap"
        return (self.rfile.read(length) if length else b""), None

    def _proxy(self) -> None:
        body, refusal = self._read_request_body()
        if refusal is not None:
            self._respond_json(
                400, {"message": refusal, "type": "invalid_request_error", "code": 400}
            )
            return
        alias = self._alias_to_normalize(body)
        payload = (
            normalize_request(body, alias, self.canonical_model)
            if alias is not None
            else body
        )
        # The rewritten body has a different size, so never reuse the inbound
        # Content-Length: http.client computes the correct one from the payload.
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in REQUEST_STRIP_HEADERS
        }
        headers["Host"] = f"127.0.0.1:{self.inner_port}"
        if alias is not None:
            headers["X-Ciru-Model-Normalized"] = alias
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.inner_port, timeout=self.read_timeout
        )
        try:
            connection.request(
                self.command, self.path, body=payload or None, headers=headers
            )
            self._relay(connection.getresponse(), alias)
        except (ConnectionError, OSError, http.client.HTTPException) as exc:
            self._respond_unavailable(exc)
        finally:
            connection.close()
            self.close_connection = True

    def _relay(self, response: http.client.HTTPResponse, alias: str | None) -> None:
        """Forward the reply with framing this proxy re-establishes for its own writes."""
        content_type = (response.getheader("Content-Type") or "").lower()
        framing = {
            key: value
            for key, value in response.getheaders()
            if key.lower() not in RESPONSE_STRIP_HEADERS
        }
        self.send_response(response.status, response.reason)
        if "text/event-stream" in content_type:
            for key, value in framing.items():
                self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            if alias is None:
                self._pump_raw(response)
            else:
                self._pump_event_stream(response, alias)
            return
        payload = response.read()
        if alias is not None:
            payload = restore_response(payload, self.canonical_pattern, alias)
        framing["Content-Length"] = str(len(payload))
        for key, value in framing.items():
            self.send_header(key, value)
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def _pump_raw(self, response: http.client.HTTPResponse) -> None:
        """Copy a response body through without inspecting it."""
        while chunk := response.read(STREAM_CHUNK_BYTES):
            self.wfile.write(chunk)
        self.wfile.flush()

    def _pump_event_stream(
        self, response: http.client.HTTPResponse, alias: str
    ) -> None:
        """Restore the caller's model ID per SSE line while streaming without delay."""
        buffered = b""
        while chunk := response.read(STREAM_CHUNK_BYTES):
            buffered += chunk
            while b"\n" in buffered:
                line, buffered = buffered.split(b"\n", 1)
                self.wfile.write(
                    restore_response(line, self.canonical_pattern, alias) + b"\n"
                )
            self.wfile.flush()
        if buffered:
            self.wfile.write(restore_response(buffered, self.canonical_pattern, alias))
            self.wfile.flush()

    def _respond_json(self, status: int, error: dict[str, object]) -> None:
        """Return a JSON error envelope with framing this proxy owns."""
        payload = json.dumps({"error": error}, separators=(",", ":")).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
        except (OSError, http.client.HTTPException):
            pass

    def _respond_unavailable(self, exc: BaseException) -> None:
        """Report an engine connection failure without pretending it is inference output."""
        self._respond_json(
            502,
            {
                "message": f"Ciru engine upstream unavailable: {exc}",
                "type": "bad_gateway",
                "code": 502,
            },
        )

    @classmethod
    def _warn(cls, message: str) -> None:
        """Emit each distinct warning once so llama-swap's log stream stays readable."""
        with cls.warned_lock:
            if message in cls.warned:
                return
            if len(cls.warned) >= MAX_WARNED_NAMES:
                cls.warned.clear()
            cls.warned.add(message)
        print(f"ciru-model-proxy: {message}", file=sys.stderr, flush=True)

    def log_message(self, _format: str, *_args: object) -> None:
        """Keep request logging in llama-swap rather than duplicating it here."""


def main() -> int:
    """Start the Ciru launcher behind the model-name-normalizing listener."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", required=True, type=int, help="llama-swap managed listen port"
    )
    parser.add_argument(
        "--entrypoint",
        default=os.environ.get("CIRU_SERVE_SCRIPT", DEFAULT_ENTRYPOINT),
        help="vendor launcher command; host and port arguments are appended",
    )
    parser.add_argument(
        "--inner-port",
        type=int,
        default=None,
        help="engine port (default: the managed port plus one, or an ephemeral port)",
    )
    parser.add_argument(
        "--canonical-model",
        default=os.environ.get("CIRU_SERVED_MODEL", DEFAULT_CANONICAL_MODEL),
        help="the single model name the vendor vLLM release actually serves",
    )
    parser.add_argument(
        "--aliases",
        default=os.environ.get("CIRU_MODEL_ALIASES", ",".join(DEFAULT_ALIASES)),
        help="comma-separated llama-swap IDs that normalize to the canonical model",
    )
    parser.add_argument("--shutdown-grace", type=float, default=30.0)
    parser.add_argument("--read-timeout", type=float, default=3600.0)
    args = parser.parse_args()

    aliases = parse_alias_list(args.aliases, args.canonical_model)
    if not aliases:
        parser.error(
            "--aliases must contain at least one name besides the canonical model"
        )
    if not args.entrypoint.strip():
        parser.error("--entrypoint must not be empty")

    inner_port = (
        args.inner_port
        if args.inner_port is not None
        else pick_inner_port(args.port + 1)
    )
    argv = launcher_argv(args.entrypoint, inner_port)
    CiruProxyHandler.inner_port = inner_port
    CiruProxyHandler.canonical_model = args.canonical_model
    CiruProxyHandler.canonical_pattern = model_field_pattern(args.canonical_model)
    CiruProxyHandler.aliases = aliases
    CiruProxyHandler.read_timeout = args.read_timeout

    # Bind the listener before spawning anything: a port clash must never leave a
    # multi-gigabyte engine running with nobody forwarding to it.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), CiruProxyHandler)
    engine = Engine(argv)

    def engine_died(code: int) -> None:
        print(
            f"ciru-model-proxy: Ciru engine exited with code {code}, closing the listener",
            file=sys.stderr,
            flush=True,
        )
        os._exit(code)

    threading.Thread(target=engine.watch, args=(engine_died,), daemon=True).start()

    def terminate(_signum: int, _frame: object) -> None:
        """Tear the whole engine process group down before this proxy exits."""
        print(
            "ciru-model-proxy: shutdown requested, stopping the Ciru engine",
            file=sys.stderr,
            flush=True,
        )
        engine.stop(args.shutdown_grace)
        os._exit(0)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    print(
        f"ciru-model-proxy: listening on 127.0.0.1:{args.port}, engine on "
        f"127.0.0.1:{inner_port}, normalizing {len(aliases)} aliases to "
        f"{args.canonical_model!r}",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        engine.stop(args.shutdown_grace)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
