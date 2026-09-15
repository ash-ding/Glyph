"""Metering model gateway: forwards pinned-model requests to Vertex, meters usage into the ledger.

Sits between the sandboxed Claude Code CLI and Vertex AI. This is the harness's authoritative
cost accounting: only the pinned/allowed model may be forwarded, every request that ultimately
succeeds is billed exactly once, and full request bodies are logged for audit.

Productionized from the spike at /tmp/glyph_v2_spikes/gateway.py (S1). Preserves two behaviours
from that spike that are load-bearing for the real deployment:

  spec delta #1 - the CLI drops the "/v1" path prefix when ANTHROPIC_VERTEX_BASE_URL is
                  overridden, so this gateway re-inserts "/v1" before forwarding upstream.
  spec delta #2 - `allowed_models` (default: {pinned_model}) is the actual enforcement point.
                  The harness also sets ANTHROPIC_SMALL_FAST_MODEL=pinned so the CLI's own
                  title/summary calls use the pinned model, but that is just cooperative
                  behaviour from the client -- this gateway does not trust it and rejects
                  anything outside allowed_models regardless.

The request handler core (`handle_request`) is a plain async function with no dependency on a
running aiohttp server, so it can be unit tested by injecting a fake `forward_fn` -- no real
Vertex call and no google.auth needed. `make_app()`/`serve_unix()` wire that core into a real
aiohttp app for production use.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Awaitable, Callable, Optional

import aiohttp
from aiohttp import web

from glyph.v2.ledger import Ledger

# Matches ".../models/<model>:streamRawPredict", ":rawPredict", or ":countTokens" alike --
# anything up to the next "/" or ":" after "/models/".
MODEL_RE = re.compile(r"/models/([^:/]+):")

ForwardFn = Callable[[str, str, dict, bytes], Awaitable[tuple]]
TokenProvider = Callable[[], str]

RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
# Transient/network-level forward errors worth a retry-then-503. A
# non-network exception (a programming bug in forward_fn) is deliberately
# NOT caught -- it must surface loudly rather than be masked as a retryable
# 503 the client retries MAX_ATTEMPTS times. (TimeoutError subclasses
# OSError; asyncio.TimeoutError is that same builtin on 3.11+.)
_RETRYABLE_FORWARD_ERRORS = (aiohttp.ClientError, asyncio.TimeoutError,
                            ConnectionError, OSError)
MAX_ATTEMPTS = 3
BASE_BACKOFF_SECONDS = 0.05  # small on purpose: keeps retry tests fast


def _default_token_provider() -> TokenProvider:
    """Build a token provider backed by google.auth default credentials.

    Imported lazily (only when the real default forward_fn is actually used) so that unit
    tests, which always inject their own forward_fn, never need google.auth installed/configured.
    """
    import google.auth
    from google.auth.transport.requests import Request as GARequest

    state = {"creds": None}

    def _token() -> str:
        if state["creds"] is None:
            state["creds"], _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        if not state["creds"].valid:
            state["creds"].refresh(GARequest())
        return state["creds"].token

    return _token


def _parse_usage(body_text: str) -> dict:
    """Extract usage counters (input/output/cache_read/cache_creation tokens) from a response.

    Handles both shapes seen from Vertex/Anthropic:
      * non-streamed: the whole body is one JSON object, usage at top level and/or under "message"
      * streamed (SSE): a series of "data: {...}\\n\\n" chunks; the final message's usage wins
    """
    usage: dict = {}

    try:
        obj = json.loads(body_text)
    except (json.JSONDecodeError, TypeError):
        obj = None
    if isinstance(obj, dict):
        if isinstance(obj.get("usage"), dict):
            usage.update(obj["usage"])
        if isinstance(obj.get("message"), dict) and isinstance(obj["message"].get("usage"), dict):
            usage.update(obj["message"]["usage"])
        if usage:
            return usage

    for line in body_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(chunk, dict):
            continue
        if isinstance(chunk.get("message"), dict) and isinstance(chunk["message"].get("usage"), dict):
            usage.update(chunk["message"]["usage"])
        if isinstance(chunk.get("usage"), dict):
            usage.update(chunk["usage"])
    return usage


class Gateway:
    """Metering gateway between the sandboxed CLI and Vertex AI."""

    def __init__(
        self,
        ledger: Ledger,
        *,
        pinned_model: str,
        allowed_models: Optional[set] = None,
        upstream_base: str = "https://aiplatform.googleapis.com",
        forward_fn: Optional[ForwardFn] = None,
        token_provider: Optional[TokenProvider] = None,
        log_path: Optional[str] = None,
    ) -> None:
        self.ledger = ledger
        self.pinned_model = pinned_model
        self.allowed_models = set(allowed_models) if allowed_models is not None else {pinned_model}
        self.upstream_base = upstream_base.rstrip("/")
        self.log_path = log_path
        self._token_provider = token_provider
        self._forward_fn = forward_fn or self._default_forward
        self._records: list = []

    @property
    def records(self) -> list:
        """Per-request metering records, in request order."""
        return self._records

    def _token(self) -> str:
        if self._token_provider is None:
            self._token_provider = _default_token_provider()
        return self._token_provider()

    async def _default_forward(self, method: str, url: str, headers: dict, body: bytes):
        """Real upstream call via aiohttp with a google.auth bearer token.

        Only reachable when no forward_fn is injected -- i.e. never in unit tests.
        """
        from aiohttp import ClientSession, ClientTimeout, TCPConnector

        hdrs = dict(headers)
        hdrs["Authorization"] = "Bearer " + self._token()
        # A long reasoning turn can stream for well over aiohttp's default
        # total=300s; capping it there raised TimeoutError mid-turn, which
        # became a silent 500 and stalled the CLI. Use no total cap, but keep
        # a sock_read guard so a genuinely dead upstream still fails.
        timeout = ClientTimeout(total=None, sock_connect=30, sock_read=300)
        async with ClientSession(connector=TCPConnector(ssl=True), timeout=timeout) as sess:
            async with sess.request(method, url, data=body, headers=hdrs) as up:
                resp_body = await up.read()
                return up.status, dict(up.headers), resp_body

    def _record(self, rec: dict) -> None:
        rec["ts"] = time.time()
        self._records.append(rec)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    async def handle_request(self, method: str, path: str, headers: dict, body: bytes):
        """Pure async core of the gateway: no aiohttp.web / server dependency.

        1. Extracts the model from `path` and enforces allowed_models (403, unbilled, unforwarded
           if not permitted).
        2. Re-inserts the "/v1" prefix if `path` lacks it, then forwards via forward_fn, retrying
           429/5xx up to MAX_ATTEMPTS times with a small backoff.
        3. Bills the ledger from the response usage ONLY on a final 2xx; a request that never
           succeeds is recorded (allowed=True, status=<last>) but bills nothing.

        Returns (status, resp_headers, resp_body_bytes).
        """
        m = MODEL_RE.search(path)
        model = m.group(1) if m else None

        if model not in self.allowed_models:
            self._record({"model": model, "path": path, "allowed": False, "status": 403})
            return 403, {}, json.dumps({"error": "model not permitted: %s" % model}).encode()

        upstream_path = path if path.startswith("/v1") else "/v1" + path
        url = self.upstream_base + upstream_path
        fwd_headers = {k: v for k, v in headers.items() if k.lower() not in ("host", "content-length")}

        status = None
        resp_headers: dict = {}
        resp_body = b""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                status, resp_headers, resp_body = await self._forward_fn(method, url, fwd_headers, body)
            except _RETRYABLE_FORWARD_ERRORS as e:  # network-level -- retry then 503, never a silent 500
                import sys
                import traceback
                print("gateway: forward attempt %d raised: %r" % (attempt, e), file=sys.stderr)
                traceback.print_exc()
                status, resp_headers, resp_body = (
                    503, {}, json.dumps({"error": "upstream forward failed: %r" % e}).encode())
            if status < 300 or status not in RETRYABLE_STATUSES or attempt == MAX_ATTEMPTS:
                break
            await asyncio.sleep(BASE_BACKOFF_SECONDS * attempt)

        if 200 <= status < 300:
            usage = _parse_usage(resp_body.decode("utf-8", "replace"))
            self.ledger.add_model_usage(
                input=int(usage.get("input_tokens", 0) or 0),
                cache_read=int(usage.get("cache_read_input_tokens", 0) or 0),
                cache_write=int(usage.get("cache_creation_input_tokens", 0) or 0),
                output=int(usage.get("output_tokens", 0) or 0),
            )
            self._record(
                {
                    "model": model,
                    "path": path,
                    "allowed": True,
                    "status": status,
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                    "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                }
            )
        else:
            self._record({"model": model, "path": path, "allowed": True, "status": status})

        return status, resp_headers, resp_body

    async def _aiohttp_handler(self, request: web.Request) -> web.Response:
        body = await request.read()
        status, resp_headers, resp_body = await self.handle_request(
            request.method, request.rel_url.path_qs, dict(request.headers), body
        )
        content_type = resp_headers.get("Content-Type", "application/json").split(";")[0].strip()
        return web.Response(status=status, body=resp_body, content_type=content_type or "application/json")

    def make_app(self) -> web.Application:
        """Build the aiohttp Application, routed for the Vertex path(s)."""
        app = web.Application(client_max_size=1024**3)
        app.router.add_route("*", "/{tail:.*}", self._aiohttp_handler)
        return app

    async def serve_unix(self, socket_path: str) -> None:
        """Production entrypoint: bind make_app() to a unix socket and serve forever."""
        import os

        app = self.make_app()
        runner = web.AppRunner(app)
        await runner.setup()
        if os.path.exists(socket_path):
            os.remove(socket_path)
        site = web.UnixSite(runner, socket_path)
        await site.start()
        while True:
            await asyncio.sleep(3600)
